# Lexical Retrieval Comparison

- Golden dataset: 19 questions
- Retrieval depth: top-5
- Both methods use the same accent-folding, stopword removal and bigram tokenizer.

## Overall

| Metric | BM25 | TF-IDF |
|---|---:|---:|
| Expected-source hit@5 | 1.000 | 0.947 |
| Mean reciprocal rank | 0.921 | 0.860 |
| Expected source at rank 1 | 0.842 | 0.789 |

## Per-question expected-source rank

| Question | Expected source | BM25 rank | TF-IDF rank |
|---|---|---:|---:|
| Shopee Việt Nam hỗ trợ những phương thức thanh toán nào? | `news/payment-methods.md` | 1 | — |
| Giới hạn giá trị đơn hàng khi thanh toán bằng Apple Pay và Google Pay trên Shopee là bao nhiêu? | `news/payment-methods.md` | 2 | 2 |
| Khi nào người mua có thể đổi phương thức thanh toán cho đơn hàng Shopee? | `news/change-payment-method.md` | 1 | 1 |
| Có thể kiểm tra phí vận chuyển và thời gian giao hàng dự kiến ở đâu trên Shopee? | `news/delivery-estimate.md` | 1 | 1 |
| Đơn đã quá thời gian dự kiến nhưng chưa giao thì người mua nên làm gì? | `news/late-delivery.md` | 1 | 1 |
| Đơn hàng quốc tế trên Shopee đi qua những kênh vận chuyển nào và ai giao khi về Việt Nam? | `news/international-shipping.md` | 1 | 1 |
| Đơn hàng quốc tế trên Shopee thường mất bao lâu để giao? | `news/international-shipping.md` | 1 | 1 |
| Người mua cần cung cấp những thông tin gì khi gửi yêu cầu Trả hàng/Hoàn tiền tại trang đơn hàng? | `news/submit-return-refund.md` | 1 | 3 |
| Yêu cầu Trả hàng/Hoàn tiền ở trạng thái Shopee đang xem xét được xử lý trong bao lâu? | `news/refund-process.md` | 1 | 1 |
| Sau khi Shopee chấp nhận phương án Trả hàng & Hoàn tiền, người mua có bao lâu để gửi trả hàng? | `news/refund-process.md` | 2 | 1 |
| Nếu chưa nhận được hàng thì người mua có phải nộp bằng chứng cho yêu cầu hoàn tiền không? | `news/refund-evidence.md` | 1 | 1 |
| Video mở kiện dùng làm bằng chứng khi hàng có vấn đề cần đáp ứng điều kiện gì? | `news/refund-evidence.md` | 1 | 1 |
| Tiền hoàn cho đơn thanh toán bằng thẻ tín dụng hoặc ghi nợ mất bao lâu và được hoàn về đâu? | `news/refund-time.md` | 1 | 1 |
| Tiền hoàn cho đơn thanh toán bằng thẻ nội địa NAPAS mất bao lâu? | `news/refund-time.md` | 2 | 2 |
| Shopee có những hình thức gửi hàng hoàn trả nào và hình thức nào miễn phí? | `news/return-shipping-methods.md` | 1 | 1 |
| Đơn không thuộc Shopee Mall tự sắp xếp trả hàng được hỗ trợ phí như thế nào? | `news/return-shipping-methods.md` | 1 | 1 |
| Theo Chính sách bảo mật Shopee, dữ liệu cá nhân là gì và chính sách áp dụng cho ai? | `legal/privacy-policy-shopee.md` | 1 | 1 |
| Theo Điều khoản dịch vụ, hợp đồng mua bán trên Shopee được ký giữa ai và ai chịu trách nhiệm? | `legal/terms-of-service-shopee.md` | 1 | 1 |
| Quy định đăng bán của Shopee nghiêm cấm những nhóm nội dung sản phẩm nào? | `legal/product-listing-regulations-shopee.md` | 1 | 1 |

## Interpretation

BM25 applies term-frequency saturation and document-length normalization. TF-IDF uses normalized term weights without BM25's saturation parameters. This report is a controlled lexical-only comparison; the production pipeline still uses BM25 by default and can select TF-IDF from the Streamlit sidebar.
