import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from vit_model import ViTModel, Config

def smooth_tensor(x, kernel_size=5):
    padding = kernel_size // 2
    kernel = torch.ones(3, 1, kernel_size, kernel_size, device=x.device) / (kernel_size * kernel_size)
    return F.conv2d(x, kernel, padding=padding, groups=3)

def project_perturbation_to_psnr(img_clean, img_adv_float, target_psnr=45.02):
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
    img_pil = Image.fromarray(img_np)
    img_pil.save(save_path, optimize=True, compress_level=9)

def run_ad_attack():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Khởi chạy C&W Attack trên {device}")
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
        
    output_dir = "submission_images"
    fail_dir = "non_sub"
    
    # Đọc danh sách ảnh từ non_sub
    if not os.path.exists(fail_dir):
        print(f"[Lỗi] Thư mục '{fail_dir}' không tồn tại. Chưa có ảnh nào bị rớt đài.")
        return
        
    non_sub_files = [f for f in os.listdir(fail_dir) if f.endswith('.png')]
    if len(non_sub_files) == 0:
        print("[Thành công] Tuyệt vời! Không còn ảnh nào trong thư mục non_sub. 100% ASR đạt được!")
        return
        
    print(f"[Data] Đội đặc nhiệm C&W đang nhắm vào {len(non_sub_files)} ảnh cứng đầu.")
    
    batch_size = 17 # Xử lý trọn gói 17 ảnh trong 1 Batch duy nhất
    num_steps = 50
    num_restarts = 25  # Tăng lên 25 lần restart kết hợp Momentum
    target_psnr_safe = 45.02 # Ép sát giới hạn PSNR để nới rộng không gian L2
    kappa = 5.0 # Margin cho C&W
    
    # Tính toán L2 norm radius tương đương với giới hạn PSNR
    max_mse_255 = (255.0 / (10 ** (target_psnr_safe / 20.0))) ** 2
    max_l2_norm = math.sqrt(max_mse_255 * 3 * 224 * 224) / 255.0
    alpha_l2 = max_l2_norm / 10.0 # Step size 10% của bán kính
    
    print(f"[L2 C&W Attack] Steps={num_steps}, Restarts={num_restarts}, Kappa={kappa}, Max L2={max_l2_norm:.4f}")
    
    total_images_processed = 0
    newly_solved = 0
    
    for batch_start in tqdm(range(0, len(non_sub_files), batch_size), desc="C&W Batch"):
        batch_end = min(batch_start + batch_size, len(non_sub_files))
        batch_filenames = non_sub_files[batch_start:batch_end]
        
        batch_clean_imgs = []
        batch_clean_tensors = []
        indices = []
        questions_batch = []
        current_idx = 0
        
        for img_filename in batch_filenames:
            # LUÔN TẢI ẢNH GỐC (CLEAN) TỪ TEST_IMG_DIR
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
                logits_clean = model(x_norm[indices_tensor], q_batch)
            Y_clean_preds = logits_clean.argmax(1).cpu().tolist()
            
        Y_batch = torch.tensor(Y_clean_preds, dtype=torch.long, device=device)
        
        best_adv_np_list = [None] * B_actual
        best_success_counts = [-1] * B_actual
        is_solved = [False] * B_actual
        
        # Multi-Restart Loop
        for restart in range(num_restarts):
            if all(is_solved): 
                break
                
            momentum_buffer = 0.0 # Khởi tạo Momentum cho mỗi lần restart
            
            if restart == 0:
                x_adv = x_clean.clone().detach().requires_grad_(True)
            else:
                # Random initialization inside L2 ball
                noise = torch.randn_like(x_clean)
                B_shape = noise.shape[0]
                noise_norms = torch.norm(noise.view(B_shape, -1), p=2, dim=1).view(B_shape, 1, 1, 1)
                random_scale = torch.rand(B_shape, 1, 1, 1, device=device)
                noise = noise / (noise_norms + 1e-8) * max_l2_norm * random_scale
                x_adv = torch.clamp(x_clean + noise, 0.0, 1.0).detach().requires_grad_(True)
                
            for step in range(num_steps):
                with torch.amp.autocast("cuda"):
                    x_norm = transforms.functional.normalize(x_adv, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    logits = model(x_norm[indices_tensor], q_batch)
                    
                    # Tính C&W Margin Loss
                    z_y = logits.gather(1, Y_batch.unsqueeze(1)).squeeze(1)
                    
                    logits_other = logits.clone()
                    logits_other.scatter_(1, Y_batch.unsqueeze(1), float('-inf'))
                    z_other = logits_other.max(dim=1)[0]
                    
                    # Chúng ta muốn tối thiểu hóa (z_y - z_other). Tương đương với C&W:
                    loss_cw = torch.clamp(z_y - z_other + kappa, min=0.0).sum()
                
                model.zero_grad()
                if x_adv.grad is not None:
                    x_adv.grad.zero_()
                loss_cw.backward()
                
                with torch.no_grad():
                    grad = x_adv.grad
                    if grad is not None:
                        grad = smooth_tensor(grad, kernel_size=5)
                        B_shape = grad.shape[0]
                        grad_norms = torch.norm(grad.view(B_shape, -1), p=2, dim=1).view(B_shape, 1, 1, 1)
                        normalized_grad = grad / (grad_norms + 1e-8)
                        
                        # Cập nhật Momentum
                        momentum_buffer = 0.9 * momentum_buffer + normalized_grad
                        
                        # Trừ gradient vì ta đang tối thiểu hóa Loss
                        x_adv = x_adv - alpha_l2 * momentum_buffer
                        
                    # L2 Projection
                    delta = x_adv - x_clean
                    delta_norms = torch.norm(delta.view(B_shape, -1), p=2, dim=1).view(B_shape, 1, 1, 1)
                    factor = torch.clamp(max_l2_norm / (delta_norms + 1e-8), max=1.0)
                    x_adv = x_clean + delta * factor
                    x_adv = torch.clamp(x_adv, 0.0, 1.0)
                    
                x_adv = x_adv.clone().detach().requires_grad_(True)
                
            x_adv_final = x_adv.detach().permute(0, 2, 3, 1).cpu().numpy()
            
            # Evaluate after PSNR projection
            batch_adv_tensors = []
            batch_adv_np_list_temp = []
            
            for i in range(B_actual):
                img_adv_np, _ = project_perturbation_to_psnr(batch_clean_imgs[i], x_adv_final[i], target_psnr=target_psnr_safe)
                batch_adv_np_list_temp.append(img_adv_np)
                
                img_adv_tensor = transforms.ToTensor()(Image.fromarray(img_adv_np))
                batch_adv_tensors.append(img_adv_tensor)
                
            x_adv_saved = torch.stack(batch_adv_tensors).to(device)
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    x_adv_saved_norm = transforms.functional.normalize(x_adv_saved, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    logits_adv = model(x_adv_saved_norm[indices_tensor], q_batch)
                Y_adv_preds = logits_adv.argmax(1).cpu().tolist()
                
            q_idx = 0
            for i in range(B_actual):
                img_filename = batch_filenames[i]
                num_q = len(image_groups[img_filename])
                
                success_count = 0
                for j in range(num_q):
                    if Y_adv_preds[q_idx + j] != Y_clean_preds[q_idx + j]:
                        success_count += 1
                q_idx += num_q
                
                # Cập nhật kết quả tốt nhất
                if not is_solved[i]:
                    if success_count == num_q:
                        is_solved[i] = True
                        best_adv_np_list[i] = batch_adv_np_list_temp[i]
                        newly_solved += 1
                    elif success_count > best_success_counts[i]:
                        best_success_counts[i] = success_count
                        best_adv_np_list[i] = batch_adv_np_list_temp[i]
                        
        # 6. Route Successfully/Unsuccessfully attacked images
        for i in range(B_actual):
            img_filename = batch_filenames[i]
            if is_solved[i]:
                save_path = os.path.join(output_dir, img_filename)
                # Xóa file cũ khỏi non_sub
                old_fail_path = os.path.join(fail_dir, img_filename)
                if os.path.exists(old_fail_path):
                    os.remove(old_fail_path)
            else:
                save_path = os.path.join(fail_dir, img_filename)
                
            executor.submit(save_image_async, best_adv_np_list[i], save_path)
            
        total_images_processed += B_actual
        print(f"[{total_images_processed}/{len(non_sub_files)}] C&W Đã bẻ khóa thêm: {newly_solved} ảnh cứng đầu!")
        
    print("\n" + "="*50)
    print("🏆 BÁO CÁO KẾT QUẢ ĐỘI ĐẶC NHIỆM C&W")
    print("="*50)
    print(f"Tổng số ảnh được giải cứu: {newly_solved} / {len(non_sub_files)}")
    print(f"Các ảnh giải cứu thành công đã được dời sang '{output_dir}'.")
    print(f"Các ảnh vẫn còn cứng đầu tiếp tục ở lại '{fail_dir}'.")
    print("="*50)
    
    print("\n[Saving] Đang chờ hoàn tất lưu toàn bộ ảnh...")
    executor.shutdown(wait=True)
    print("[Saving] Đã hoàn tất!")

if __name__ == "__main__":
    run_ad_attack()
