# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603752
- **Họ và tên:** Nguyễn Tiến Đạt
- **SEED** (in ở ô đầu notebook): 5659

## 1. Bằng chứng

| Tệp                                                                  | Có? |
| -------------------------------------------------------------------- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output  | x   |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | x   |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước          | x   |

## 2. Kết quả chính

| Model                         | Precision | Recall | F1 (test) | F1 (SUB 40) | train (s) | tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| ----------------------------- | --------- | ------ | --------- | ----------- | --------- | ------------- | ---------- | ---------- | ------------ |
| 1. BoW + LogReg (tự code)     | 0.986     | 0.753  | 0.854     | 0.788       | 12.555    | 5360.287      | 0.048      | 0.249      | 0.0          |
| 2. BoW + MLP (deep learning)  | 0.978     | 0.907  | 0.941     | 0.889       | 14.709    | 26399.341     | 0.287      | 0.624      | 0.0          |
| 3a. Encoder (MiniLM) + kNN    | 0.802     | 0.959  | 0.873     | 0.974       | 80.130    | 37.203        | 38.993     | 59.995     | 0.0          |
| 3b. Encoder (MiniLM) + LogReg | 0.939     | 0.959  | 0.949     | 0.974       | 80.521    | 46.400        | 22.653     | 31.179     | 0.0          |
| 3c. Encoder (MiniLM) + MLP    | 0.968     | 0.948  | 0.958     | 0.974       | 81.041    | 43.180        | 23.415     | 40.492     | 0.0          |
| 4a. LLM nhỏ zero-shot         | 0.864     | 0.950  | 0.905     | 0.905       | 0.000     | 1.137         | 402.956    | 2474.818   | 0.0          |
| 4b. LLM lớn zero-shot         | 0.950     | 0.950  | 0.950     | 0.950       | 0.000     | 1.116         | 401.115    | 2504.952   | 0.0          |
| 4c. LLM nhỏ few-shot          | 0.952     | 1.000  | 0.976     | 0.976       | 0.000     | 0.903         | 530.118    | 3062.574   | 0.0          |

**Ghi chú:**

- Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
- SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
- Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
- train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

### 1. `ml_f1` — Dự đoán F1 của BoW + LogReg

**Dự đoán:** F1 ~0.7, precision cao hơn vì với ngưỡng 0.5, khả năng số lượng text dự đoán đúng sẽ cao

**Thực tế:** F1 = 0.854, precision = 0.986, recall = 0.753

**Phân tích lệch:** Dự đoán F1 thấp hơn thực tế (~15% chênh lệch). Đánh giá thấp hiệu quả của BoW. Dữ liệu sẵn có chứa nhiều các từ khóa spam nên score tương đối cao.

### 2. `dl_relu` — Loại bỏ ReLU trong MLP

**Dự đoán:** Tương đương với mô hình logistic regression. F1 giảm

**Thực tế:** Không phải tương đương với LogReg đơn thuần. MLP không ReLU vẫn có thể học được quyết định phi tuyến qua nhiều phép chiếu. F1 thực tế giảm từ 0.941 xuống 0.88-0.90.

**Phân tích lệch:** nhầm lẫn về kiến trúc. Nhiều tầng linear xếp chồng học đặc trưng tốt hơn LogReg một tầng. Không có ReLU không ảnh hưởng quá lớn về LogReg.

### 3. `llm_fewshot` — Ảnh hưởng của few-shot learning

**Dự đoán:** F1 tăng. Tăng khoảng 1.5 lần (từ ~0.9 lên ~1.35, vô lý vì F1 max = 1.0)

**Thực tế:** F1 tăng từ 0.905 (zero-shot) lên 0.976 (few-shot) — tăng 7.8%, không phải 50%

**Phân tích lệch:** Đánh giá không đúng, thêm 4 texts không làm tăng đáng kể chi phí đầu vào, mà chi phí đầu vào lại tương đối rẻ so với chi phí đầu ra. Trong khi đó `max_tokens` đã giới hạn sẵn kích thước đầu ra.

## 4. Phân tích 50 case khó

_Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai._

## 5. Thử thách cá nhân

_Đề theo SEED của bạn, cách làm, kết quả, giải thích._

## 6. Phương án cho sản phẩm thật

_Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì._

## 7. AI log và điều đã học

Thực hành luôn là cách tốt nhất để hiểu rõ một lĩnh vực, cộng thêm áp lực thời gian có thể thúc đẩy giới hạn học tập của bản thân.
