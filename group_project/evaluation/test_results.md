# Test Results

- Thời điểm chạy: 2026-08-04 (Asia/Saigon)
- Python: 3.12.13
- Lượt nền trước khi thêm provider adapter: `unittest` **49/49 passed**.
- Lượt chốt sau toàn bộ thay đổi: `pytest tests -q -rs` — **51/51 passed**,
  0 failed, 0 skipped (101.26 giây).

## Phạm vi

| Nhóm kiểm tra | Nội dung |
|---|---|
| Task 1-3 | Landing data, metadata, standardized Markdown và kích thước nội dung |
| Task 4-5 | Chunking, cấu hình, BGE-M3 semantic search, format và thứ tự score |
| Task 6-9 | BM25, reranking/RRF, PageIndex live fallback và retrieval pipeline |
| Task 10 | Context reordering, citation context và generation live qua provider đã cấu hình |
| Bonus lexical | TF-IDF ranking, accent-insensitive query và tích hợp vào hybrid pipeline |
| UI security | Highlight tiếng Việt, merge evidence và escape HTML trước khi render `<mark>` |
| Provider | Generic OpenAI-compatible config, model override và fallback cấu hình cũ |

## Live integration

- PageIndex trả về danh sách có marker `source=pageindex`.
- LLM provider trả lời được request kiểm tra và Task 10 tạo answer dictionary có citation.
- Streamlit end-to-end trả lời đúng câu hỏi thời hạn, hiển thị 5 nguồn, dense
  similarity, BM25 rank và các thẻ `<mark>` evidence; console không có lỗi.
- RAGAS chạy đủ 19 câu x 2 cấu hình; mỗi cấu hình có 19 hàng. Mỗi cấu hình có
  1/76 giá trị metric không parse được từ judge và được giữ là `N/A`, không tự
  thay bằng điểm giả. Xem `results.md` để biết aggregate và worst performers.

## Ghi chú tái lập

Môi trường local hiện dùng Python runtime 3.12.13 với packages trong `.venv`.
Các test live cần `LLM_*` và `PAGEINDEX_API_KEY`; nếu không có key hợp lệ, hai
test tích hợp tương ứng có thể được skip theo thiết kế, còn unit/regression test
vẫn chạy độc lập.
