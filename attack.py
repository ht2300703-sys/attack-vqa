import os
import json
import csv
import math
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from vit_model import ViTModel, Config

# =============================================================================
# 🛠️ 1. HELPER FUNCTIONS FOR PSNR & PROJECTION
# =============================================================================

def smooth_tensor(x, kernel_size=5):
    """
    Applies a simple box blur (average pooling filter) on a batch of tensors.
    Useful for smoothing high-frequency adversarial gradients or perturbations.
    """
    padding = kernel_size // 2
    kernel = torch.ones(3, 1, kernel_size, kernel_size, device=x.device) / (kernel_size * kernel_size)
    return F.conv2d(x, kernel, padding=padding, groups=3)

def project_perturbation_to_psnr(img_clean, img_adv_float, target_psnr=45.05):
    """
    Finds the maximum scaling factor gamma in [0.0, 1.0] for the perturbation
    such that the quantized integer image has PSNR >= target_psnr.
    Returns the final quantized adversarial image as a numpy uint8 array.
    """
    img_clean_f64 = img_clean.astype(np.float64)
    delta = (img_adv_float * 255.0) - img_clean_f64
    max_mse = (255.0 / (10 ** (target_psnr / 20.0))) ** 2
    
    low = 0.0
    high = 1.0
    best_gamma = 0.0
    best_img = img_clean.copy()
    
    for _ in range(20):
        mid = (low + high) / 2.0
        candidate_img = np.round(img_clean_f64 + mid * delta)
        candidate_img = np.clip(candidate_img, 0, 255)
        
        mse = np.mean((candidate_img - img_clean_f64) ** 2)
        if mse <= max_mse:
            best_gamma = mid
            best_img = candidate_img
            low = mid 
        else:
            high = mid
            
    return best_img.astype(np.uint8), best_gamma

def save_image_async(img_np, save_path):
    """Helper to save image with max PNG compression in a background thread."""
    img_pil = Image.fromarray(img_np)
    img_pil.save(save_path, optimize=True, compress_level=9)

# =============================================================================
# 🚀 2. MAIN L2-PGD ADVERSARIAL ATTACK EXECUTION 
# =============================================================================

def run_attack():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Khởi chạy tấn công trên {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    executor = ThreadPoolExecutor(max_workers=8)
    
    vocab = torch.load(Config.VOCAB_PATH, weights_only=True)
    ans_vocab = torch.load(Config.ANS_PATH, weights_only=True)
    
    def encode_q(q_str: str) -> torch.Tensor:
        tokens = q_str.lower().replace("?", "").split()
        ids = [vocab.get(w, 1) for w in tokens]
        ids = ids[:Config.MAX_LEN]
        ids += [0] * (Config.MAX_LEN - len(ids))
        return torch.tensor(ids, dtype=torch.long)
    
    model = ViTModel(
        vocab_size=len(vocab),    num_answer=len(ans_vocab),
        d_model=Config.D_MODEL,   n_heads=Config.N_HEADS,
        n_layers=Config.N_LAYERS, ff_dim=Config.FF_DIM,
        dropout=Config.DROPOUT,   max_q_len=Config.MAX_LEN,
    ).to(device)
    
    print(f"[Model] Load weights từ: {Config.SAVE_PATH}")
    model.load_state_dict(torch.load(Config.SAVE_PATH, map_location=device, weights_only=True), strict=False)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
        
    with open(Config.TEST_JSON, "r") as f:
        questions_list = json.load(f)["questions"]
        
    image_groups = {}
    for q_item in questions_list:
        img_name = q_item["image_filename"]
        if img_name not in image_groups:
            image_groups[img_name] = []
        image_groups[img_name].append(q_item)
        
    all_unique_images = list(image_groups.keys())
    
    # 4. Hyperparameters cho L_inf PGD Attack (Box Blur 5x5)
    batch_size = 12         
    num_steps = 15          
    epsilon = 10.0 / 255.0
    alpha = 0.6 / 255.0
    target_psnr_safe = 45.02
    
    print(f"[Attack] Sử dụng L_inf PGD với Box Blur 5x5 (epsilon={epsilon:.4f}, alpha={alpha:.4f}, steps={num_steps})")

    output_dir = "submission_images"
    fail_dir = "non_sub"
    
    for d in [output_dir, fail_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    
    unique_images = all_unique_images
    
    total_questions = 0
    successful_attacks = 0
    successful_images = 0
    total_images_processed = 0
    psnr_values = []
    
    criterion = nn.CrossEntropyLoss()
    
    for batch_start in tqdm(range(0, len(unique_images), batch_size), desc="Tấn công Batch"):
        batch_end = min(batch_start + batch_size, len(unique_images))
        batch_filenames = unique_images[batch_start:batch_end]
        
        batch_clean_imgs = []
        batch_clean_tensors = []
        indices = []
        questions_batch = []
        current_idx = 0
        
        for img_filename in batch_filenames:
            img_path = os.path.join(Config.TEST_IMG_DIR, img_filename)
            img_pil = Image.open(img_path).convert("RGB")
            img_np = np.array(img_pil)
            batch_clean_imgs.append(img_np)
            
            img_tensor = transforms.ToTensor()(img_pil)
            batch_clean_tensors.append(img_tensor)
            
            q_items = image_groups[img_filename]
            for q_item in q_items:
                questions_batch.append(q_item)
                indices.append(current_idx)
            current_idx += 1
            
        B_actual = len(batch_clean_imgs)
        K_actual = len(questions_batch)
        
        x_clean = torch.stack(batch_clean_tensors).to(device) 
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device) 
        q_batch = torch.stack([encode_q(q_item["question"]) for q_item in questions_batch]).to(device) 
        
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                x_norm = transforms.functional.normalize(x_clean, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                x_norm_batched = x_norm[indices_tensor]
                logits_clean = model(x_norm_batched, q_batch)
            Y_clean_preds = logits_clean.argmax(1).cpu().tolist()
            
        x_adv = x_clean.clone().detach().requires_grad_(True)
        Y_batch = torch.tensor(Y_clean_preds, dtype=torch.long, device=device)
        
        for step in range(num_steps):
            with torch.amp.autocast("cuda"):
                x_norm = transforms.functional.normalize(x_adv, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                x_norm_batched = x_norm[indices_tensor]
                logits = model(x_norm_batched, q_batch)
                loss = criterion(logits, Y_batch)
            
            final_loss = loss.item()
            
            model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            loss.backward()
            
            with torch.no_grad():
                grad = x_adv.grad
                if grad is not None:
                    grad = smooth_tensor(grad, kernel_size=5) 
                    x_adv = x_adv + alpha * grad.sign()
                    
                # Projection (L_inf ball clamp)
                x_adv = torch.max(x_clean - epsilon, torch.min(x_adv, x_clean + epsilon))
                # Value clamp [0.0, 1.0]
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
                
            x_adv = x_adv.clone().detach().requires_grad_(True)
            
        x_adv_final = x_adv.detach().permute(0, 2, 3, 1).cpu().numpy() 
        
        # 4. Binary Search Projection for each unique image (CPU/Numpy)
        batch_adv_tensors = []
        batch_adv_np_list = []
        for i in range(B_actual):
            img_np = batch_clean_imgs[i]
            x_adv_img = x_adv_final[i]
            
            img_adv_np, gamma = project_perturbation_to_psnr(img_np, x_adv_img, target_psnr=target_psnr_safe)
            batch_adv_np_list.append(img_adv_np)
            
            img_adv_pil_mem = Image.fromarray(img_adv_np)
            img_adv_tensor = transforms.ToTensor()(img_adv_pil_mem)
            batch_adv_tensors.append(img_adv_tensor)
            
            mse = np.mean((img_adv_np.astype(np.float64) - img_np.astype(np.float64)) ** 2)
            psnr = float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
            psnr_values.append(psnr)
            
        # 5. Evaluate final model predictions on saved integer images in batch
        x_adv_saved = torch.stack(batch_adv_tensors).to(device) 
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                x_adv_saved_norm = transforms.functional.normalize(x_adv_saved, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                x_adv_saved_norm_batched = x_adv_saved_norm[indices_tensor]
                logits_adv = model(x_adv_saved_norm_batched, q_batch)
            Y_adv_preds = logits_adv.argmax(1).cpu().tolist()
            
        # 6. Route Successfully/Unsuccessfully attacked images
        q_idx = 0
        for i in range(B_actual):
            img_filename = batch_filenames[i]
            num_q = len(image_groups[img_filename])
            
            is_success = True
            for j in range(num_q):
                if Y_adv_preds[q_idx + j] == Y_clean_preds[q_idx + j]:
                    is_success = False
                else:
                    successful_attacks += 1
            q_idx += num_q
            
            if is_success:
                successful_images += 1
                save_path = os.path.join(output_dir, img_filename)
            else:
                save_path = os.path.join(fail_dir, img_filename)
                
            executor.submit(save_image_async, batch_adv_np_list[i], save_path)
                
        total_questions += K_actual
        total_images_processed += B_actual
        
        current_asr = (successful_attacks / total_questions) * 100
        valid_psnrs = [p for p in psnr_values if p != float('inf')]
        avg_psnr = np.mean(valid_psnrs) if valid_psnrs else 45.0
        print(f"[{total_images_processed}/{len(unique_images)}] Unique images attacked | ASR: {current_asr:.2f}% ({successful_attacks}/{total_questions}) | Avg PSNR: {avg_psnr:.2f} dB | Loss: {final_loss:.4f}")
        
    final_asr = (successful_attacks / total_questions) * 100
    print("\n" + "="*50)
    print("🏆 BÁO CÁO KẾT QUẢ TẤN CÔNG (L2-PGD)")
    print("="*50)
    print(f"Tổng số ảnh thực hiện: {total_images_processed}")
    print(f"Tổng số câu hỏi:       {total_questions}")
    print(f"Số lượng câu bị lừa:   {successful_attacks}")
    print(f"Tỉ lệ lừa (Question ASR): {final_asr:.2f}%")
    print(f"Số lượng ảnh thành công hoàn toàn: {successful_images}/{total_images_processed} -> Thư mục '{output_dir}'")
    print(f"Số lượng ảnh bị lỗi một phần/hoàn toàn: {total_images_processed - successful_images}/{total_images_processed} -> Thư mục '{fail_dir}'")
    
    valid_psnrs = [p for p in psnr_values if p != float('inf')]
    avg_psnr = np.mean(valid_psnrs) if valid_psnrs else 45.0
    min_psnr = np.min(valid_psnrs) if valid_psnrs else 45.0
    print(f"PSNR trung bình (perturbed): {avg_psnr:.2f} dB")
    print(f"PSNR nhỏ nhất (perturbed):   {min_psnr:.2f} dB")
    print("="*50)
    
    print("\n[Saving] Đang chờ hoàn tất lưu toàn bộ ảnh adversarial bằng ThreadPool đa nhân...")
    executor.shutdown(wait=True)
    print("[Saving] Đã hoàn tất lưu toàn bộ ảnh phân loại!")

if __name__ == "__main__":
    run_attack()
