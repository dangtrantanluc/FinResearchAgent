# FinResearch Agent

Công cụ nghiên cứu tài chính cho doanh nghiệp niêm yết Việt Nam: nhận một câu hỏi, trả về báo cáo có số liệu tính bằng code và nhận định có trích dẫn. Repo có đủ các phần: dữ liệu, ML, tool số liệu, tìm kiếm trên báo cáo thường niên và luồng agent chạy với Gemini.

## Đã có

| Thành phần | File | Việc nó làm |
|---|---|---|
| Trích xuất | `finresearch/extract.py` | Đọc bảng HTML trong file OCR, tìm cột mã số Thông tư 200, tách ba báo cáo tài chính |
| Kiểm tra và chọn báo cáo | `finresearch/build.py` | Kiểm tra đẳng thức kế toán, loại ngân hàng / chứng khoán / bảo hiểm, chọn một báo cáo cho mỗi mã–năm |
| Đặc trưng | `finresearch/panel.py` | 53 chỉ số cho mỗi công ty–năm, nhãn tăng trưởng doanh thu năm sau, Beneish M-Score |
| Train | `notebooks/train_growth_model.ipynb` | XGBoost, baseline, backtest theo thời gian, khoảng dự báo conformal, SHAP và độ ổn định |
| Cơ sở dữ liệu | `finresearch/db.py`, `scripts/load_db.py` | Nạp số liệu, chỉ số và dự báo vào Postgres |
| Tools | `finresearch/tools/financials.py` | Số liệu, ratios, so sánh với ngành và dự báo; mỗi con số có ID và nguồn |
| Tách báo cáo | `finresearch/rag/ingest.py` | Đọc PDF theo đúng thứ tự cột, bỏ phần lặp lại trên mọi trang, chunk không vắt qua trang |
| Tìm kiếm | `finresearch/rag/search.py` | Vector (bge-m3) kết hợp từ khóa (TF-IDF), gộp bằng RRF; rerank bằng cross-encoder là tùy chọn |
| Agent | `finresearch/agent/` | Luồng LangGraph: đọc câu hỏi, lấy số, tìm đoạn trích, viết nháp, kiểm tra từng nhận định, dựng báo cáo |
| Giao diện | `app.py` | Streamlit: báo cáo, nhận định bị loại, số liệu đã dùng, biểu đồ SHAP, đoạn trích |

## Dữ liệu

[`vduydong/ocr_annual_financials`](https://huggingface.co/datasets/vduydong/ocr_annual_financials): bản OCR của 18.231 BCTC kiểm toán năm 2015–2025, giấy phép CC BY-NC 4.0. Dữ liệu không nằm trong repo; script tự tải về `data/raw/` (khoảng 3 GB).

`reference/companies.csv` (sàn, ngành ICB của từng mã) lấy từ thư viện vnstock.

## Chạy trên máy

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/build_dataset.py        # tải + trích, ghi data/processed/*.parquet
.venv/bin/python -m pytest -q
.venv/bin/python notebooks/train_growth_model.py # train, ghi artifacts/
```

## Cơ sở dữ liệu và tools

```bash
docker compose up -d --wait                              # Postgres + pgvector, cổng 5433
.venv/bin/python scripts/load_db.py                      # nạp lại toàn bộ bảng, khoảng 10 giây
.venv/bin/python -m finresearch.tools.financials FPT 2022 2025
```

Bốn tool trả về danh sách `Metric`. Mỗi metric có ID dạng `FPT.revenue.2024`, giá trị, đơn vị và nguồn; agent sẽ để LLM tham chiếu ID thay vì viết số.

| Tool | Trả về |
|---|---|
| `get_financials(ticker, start_year, end_year)` | 16 dòng chính của ba báo cáo, đơn vị đồng |
| `get_ratios(ticker, start_year, end_year)` | Tăng trưởng, biên lợi nhuận, ROE, đòn bẩy, vòng quay, chất lượng dòng tiền, CAGR |
| `get_peer_stats(ticker, year)` | Trung vị ngành và vị trí của công ty trong ngành |
| `get_forecast(ticker)` | Tăng trưởng và doanh thu dự báo năm tới kèm khoảng 80% |

`reference/scope_changes.csv` ghi các thay đổi phạm vi hợp nhất mà mô hình không thể biết. Hiện có một dòng: từ 01/01/2026 FPT không còn hợp nhất FPT Telecom, nên `get_forecast("FPT")` trừ doanh thu của FOX khỏi doanh thu cơ sở 2025 trước khi áp tăng trưởng.

## Tìm kiếm trên báo cáo thường niên

Đặt PDF vào `data/reports/<MÃ>/<MÃ>_<NĂM>.pdf` (ví dụ `data/reports/FPT/FPT_2024.pdf`), rồi:

```bash
.venv/bin/python scripts/ingest_reports.py               # tách, embed, lưu vào Postgres
.venv/bin/python -m finresearch.rag.search "Vì sao doanh thu FPT tăng năm 2024?" --ticker FPT --year 2024
.venv/bin/python scripts/eval_retrieval.py               # chấm trên eval/retrieval_fpt.jsonl
```

Mỗi kết quả có ID dạng `FPT-AR2024-p037-c1` và số trang của file PDF, để báo cáo trích dẫn được `[BCTN FPT 2024, tr. 37]`. Có ba loại chunk:

- **Văn bản:** trang đọc theo cột, từ trên xuống.
- **Bảng:** trang có lẫn bảng hoặc biểu đồ được đọc thêm một lần theo hàng ngang, để mỗi hàng giữ nhãn đi cùng số (ví dụ `Tổng | 52.618 | 61.850 | 17,5%`).
- **Báo cáo tài chính:** trang gần như toàn số không tham gia tìm kiếm, vì số liệu lấy từ cơ sở dữ liệu.

Model `BAAI/bge-m3` chạy trên máy, tải về lần đầu khoảng 2,3 GB. Chế độ `--mode hybrid_rerank` tải thêm `BAAI/bge-reranker-v2-m3` (2,3 GB).

### Kết quả tìm kiếm

20 câu hỏi về 5 báo cáo thường niên của FPT (2021–2025), một nửa bằng tiếng Anh, mỗi câu gắn với các trang chứa đáp án. Một câu tính là trúng khi có ít nhất một trong 6 kết quả đầu nằm đúng trang.

| Chế độ | Recall@6 | MRR | Recall câu tiếng Anh | Recall câu tiếng Việt |
|---|---|---|---|---|
| Chỉ từ khóa | 0,70 | 0,53 | 0,40 | 1,00 |
| Chỉ vector | 1,00 | 0,73 | 1,00 | 1,00 |
| Hybrid (mặc định) | 1,00 | 0,75 | 1,00 | 1,00 |
| Hybrid + rerank | 0,95 | 0,82 | 0,90 | 1,00 |

Tìm theo từ khóa hỏng với câu hỏi tiếng Anh vì tài liệu bằng tiếng Việt; vector xử lý được. Rerank đưa đáp án lên vị trí đầu thường hơn nhưng làm rớt một câu khỏi top 6, nên không bật mặc định. Bộ câu hỏi nhỏ và do chính người viết hệ thống soạn từ cùng tài liệu (`scripts/make_retrieval_eval.py`), nên đây là phép thử để phát hiện hỏng hóc chứ chưa phải thước đo chất lượng.

## Agent

```bash
cp .env.example .env                                     # điền GEMINI_API_KEY (lấy ở Google AI Studio)
.venv/bin/python -m finresearch.llm                      # thử kết nối tới Gemini
.venv/bin/python -m finresearch.agent.graph "Phân tích FPT giai đoạn 2022–2025 và ước tính doanh thu 2026"
.venv/bin/streamlit run app.py
```

Luồng cố định gồm bảy bước: `intent → gather → plan_queries → retrieve → draft → validate → render`. Mô hình ngôn ngữ chỉ làm hai việc: viết nháp và chấm xem đoạn trích có hỗ trợ nhận định không.

Báo cáo được viết dưới dạng danh sách nhận định. Trong nhận định, số liệu chỉ xuất hiện dưới dạng `{{FPT.revenue.2024}}`; bước cuối điền giá trị từ cơ sở dữ liệu. Trước khi tới người đọc, mỗi nhận định phải qua các kiểm tra sau:

| Kiểm tra | Loại nhận định khi |
|---|---|
| Tham chiếu số liệu | ID trong `{{...}}` không tồn tại |
| Số gõ tay | Có con số không phải tham chiếu, không phải năm, và không in trong đoạn trích được dẫn |
| Chiều tăng giảm | Viết "tăng" cho một tỷ lệ tăng trưởng âm, hoặc viết "giảm" giữa hai số của cùng một chỉ tiêu trong khi năm sau cao hơn năm trước (và ngược lại) |
| Nguồn | Dẫn một đoạn trích không tồn tại, hoặc không có cả số liệu lẫn đoạn trích |
| Hỗ trợ (cần mô hình) | Mô hình chấm rằng đoạn trích không nói điều nhận định nói, hoặc một con số bị dùng sai nghĩa (ví dụ tốc độ tăng trưởng bị viết thành tỷ trọng) |

Ở bước chấm, mô hình nhìn thấy mỗi con số kèm ý nghĩa của nó, ví dụ `⟦16,8% = CAGR doanh thu 2022–2025⟧`, và được dặn không tìm các số này trong đoạn trích.

**Chọn model:** `GEMINI_MODEL_STRONG` (viết nháp) và `GEMINI_MODEL_FAST` (chấm) trong `.env` là danh sách model cách nhau bằng dấu phẩy; model đầu tiên trả lời được sẽ được dùng. Giao diện ghi rõ model nào đã trả lời.

**Hạn mức của key miễn phí** (theo lỗi API trả về ngày 08/10/2026): 20 lượt gọi mỗi ngày cho mỗi model, và không có hạn mức cho các model Pro. Mỗi báo cáo tốn 2 lượt, một lượt viết nháp và một lượt chấm, nên hai danh sách mặc định dùng các model khác nhau: `gemini-3.5-flash` rồi `gemini-3-flash-preview` để viết, `gemini-3.5-flash-lite` rồi `gemini-3.1-flash-lite` để chấm. Các model Flash mới nhất hay báo quá tải (lỗi 503) nên không nằm trong mặc định.

### Kết quả kiểm tra báo cáo

Năm câu hỏi cố định chạy qua toàn bộ luồng với Gemini (`scripts/eval_reports.py`); báo cáo sinh ra nằm trong `eval/reports/`.

| Câu hỏi | Nhận định | Giữ lại | Loại bởi kiểm tra cứng | Loại bởi model chấm | Có trích dẫn | Được hỗ trợ đầy đủ | Thời gian |
|---|---|---|---|---|---|---|---|
| FPT 2022–2025, tiếng Việt | 19 | 19 | 0 | 0 | 11 | 10 | 35 giây |
| FPT 2022–2025, tiếng Anh | 21 | 21 | 0 | 0 | 9 | 9 | 50 giây |
| HPG 2023–2025 (kho không có BCTN) | 18 | 17 | 1 | 0 | 0 | — | 31 giây |
| MWG 2022–2025, tiếng Anh (kho không có BCTN) | 17 | 17 | 0 | 0 | 0 | — | 69 giây |
| So sánh FPT và ELC | 22 | 22 | 0 | 0 | 5 | 5 | 30 giây |

Nhận định duy nhất bị loại dẫn tới một số liệu không tồn tại (dòng tiền kinh doanh 2025 của HPG, vốn đã bị loại ở bước kiểm tra đẳng thức kế toán).

Tỷ lệ giữ lại cao không chứng minh được bộ kiểm tra có tác dụng, nên có thêm phép thử gài nhận định (`scripts/eval_judge.py`): 3 nhận định đúng và 5 nhận định sai (nguyên nhân bịa, nói ngược tài liệu, số liệu dùng sai nghĩa, khẳng định vượt ra ngoài số liệu, rủi ro không có trong đoạn trích). Model chấm `gemini-3.5-flash-lite` xử lý đúng cả 8. Cả hai phép thử đều nhỏ và mới chạy một lần.

**Không có API key** thì luồng vẫn chạy: bản nháp được viết theo mẫu câu cố định từ số liệu, các đoạn trích tìm được in nguyên văn ở cuối, và báo cáo ghi rõ chưa có nhận định định tính. Chế độ này cũng là thứ các test dùng.

## Train trên Kaggle

1. Tạo notebook mới, chọn *File → Import Notebook*, tải lên `notebooks/train_growth_model.ipynb`.
2. Trong *Settings*, bật **Internet**. Không cần GPU.
3. *Run All*. Lần đầu mất khoảng 10–15 phút, phần lớn là tải và trích dữ liệu.
4. Tải thư mục `artifacts/` ở tab *Output* về, đặt vào `artifacts/` của repo.

Notebook tự chứa mã nguồn của package `finresearch`. Khi sửa code trong repo, sinh lại notebook bằng `python scripts/make_notebook.py`.

## Kết quả hiện tại

Số liệu dưới đây là của lần train trên Kaggle (xgboost 3.4.1); model và các file kết quả của lần đó nằm trong `artifacts/`.

Nhãn là log tăng trưởng doanh thu năm sau, trừ trung vị của mọi công ty trong cùng năm. Test trên hai năm mô hình chưa thấy (2023→2024, 2024→2025), 1.929 công ty–năm.

| Mô hình | MAE | So với "bằng mặt bằng chung" | IC (tương quan hạng) |
|---|---|---|---|
| Bằng mặt bằng chung | 0,207 | — | — |
| Giữ nguyên đà năm nay | 0,296 | −42,5% | 0,04 |
| Ridge | 0,210 | −1,2% | 0,17 |
| XGBoost | 0,201 | +2,9% | 0,27 |

Mô hình xếp hạng được công ty tăng nhanh / chậm nhưng không thu hẹp được sai số bao nhiêu: tăng trưởng doanh thu một năm tới phần lớn không nằm trong BCTC năm nay. Khoảng dự báo 80% cho nhóm ổn định nhất vẫn rộng cỡ −21% đến +26% quanh điểm dự báo.

## Giới hạn

- Số liệu từ OCR. Dòng nào không khớp đẳng thức kế toán thì bị loại, nhưng lỗi ở dòng không có đẳng thức kiểm tra vẫn có thể lọt.
- Bộ dữ liệu thiếu một số mã (ví dụ CMG) và mô hình không dùng ngân hàng, chứng khoán, bảo hiểm.
- Ngưỡng M-Score −1,78 được ước lượng trên doanh nghiệp Mỹ; ở đây chỉ dùng để xếp hạng trong ngành.
- Không có số liệu quý: dữ liệu quý của vnstock bản miễn phí có nhãn kỳ không khớp nội dung, nên không dùng.
