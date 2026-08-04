"""
RAG Chatbot — E-commerce Support (Streamlit UI/UX)
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10)
Tích hợp trực quan hóa hiện tượng "Lost in the Middle" và kiểm tra trích dẫn nguồn.

Chạy:
    streamlit run app.py
"""

import os
import sys
import re
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Tải cấu hình môi trường
load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# KIỂM TRA HỆ THỐNG BACKEND & THIẾT LẬP MOCKUP FALLBACK
# =============================================================================

try:
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import (
        reorder_for_llm, 
        format_context, 
        LLM_MODEL, 
        SYSTEM_PROMPT, 
        TEMPERATURE, 
        TOP_P
    )
    HAS_BACKEND = True
except Exception as e:
    HAS_BACKEND = False
    # Mockup metadata và prompt cho chế độ độc lập/fallback
    LLM_MODEL = "mock-model"
    SYSTEM_PROMPT = ""
    TEMPERATURE = 0.3
    TOP_P = 0.9

# Dữ liệu mockup phục vụ chạy thử nghiệm khi chưa có API Key hoặc chạy độc lập
MOCK_DOCUMENTS = {
    "thanh toán": [
        {
            "content": "Shopee hỗ trợ các phương thức thanh toán sau: Thẻ tín dụng/ghi nợ (Visa, Mastercard, JCB), Ví ShopeePay, dịch vụ mua trước trả sau SPayLater, Thanh toán khi nhận hàng (COD), và Chuyển khoản ngân hàng trực tuyến.",
            "metadata": {"source": "Shopee Payment Policy.pdf", "title": "Chính Sách Thanh Toán Shopee", "type": "legal", "year": 2026},
            "score": 0.95
        },
        {
            "content": "Để thay đổi phương thức thanh toán của đơn hàng đã đặt, người mua phải tiến hành hủy đơn hàng hiện tại và đặt lại đơn hàng mới với phương thức mong muốn, với điều kiện đơn hàng chưa chuyển sang trạng thái chờ vận chuyển.",
            "metadata": {"source": "Order Management Guide.md", "title": "Hướng Dẫn Quản Lý Đơn Hàng", "type": "news", "year": 2025},
            "score": 0.88
        },
        {
            "content": "Hạn mức thanh toán giao dịch bằng Ví ShopeePay là tối đa 50.000.000 VND mỗi ngày nhằm tăng cường bảo mật tài khoản cho khách hàng.",
            "metadata": {"source": "ShopeePay Security Terms.pdf", "title": "Điều Khoản Bảo Mật ShopeePay", "type": "legal", "year": 2026},
            "score": 0.76
        },
        {
            "content": "Thanh toán qua dịch vụ SPayLater cho phép người mua trả góp với các kỳ hạn linh hoạt 1, 3, 6 hoặc 12 tháng, với mức phí quản lý và lãi suất minh bạch theo quy định.",
            "metadata": {"source": "SPayLater Terms.pdf", "title": "Điều Khoản Dịch Vụ SPayLater", "type": "legal", "year": 2026},
            "score": 0.69
        },
        {
            "content": "Mọi thông tin thẻ ngân hàng quốc tế liên kết trên Shopee đều được mã hóa và bảo mật nghiêm ngặt theo tiêu chuẩn quốc tế PCI-DSS cấp độ cao nhất.",
            "metadata": {"source": "Global Security Standards.md", "title": "Tiêu Chuản Bảo Mật Quốc Tế", "type": "news", "year": 2024},
            "score": 0.61
        }
    ],
    "đổi trả": [
        {
            "content": "Thời hạn yêu cầu trả hàng và hoàn tiền trên Shopee là trong vòng 15 ngày kể từ ngày giao hàng thành công đối với tất cả các sản phẩm thuộc Shopee Mall.",
            "metadata": {"source": "Shopee Mall Refund Policy.pdf", "title": "Chính Sách Hoàn Tiền Shopee Mall", "type": "legal", "year": 2026},
            "score": 0.97
        },
        {
            "content": "Đối với các sản phẩm không thuộc Shopee Mall, thời hạn khiếu nại mặc định là 3 ngày kể từ ngày đơn hàng hoàn thành hoặc hiển thị trạng thái giao hàng thành công.",
            "metadata": {"source": "Standard Refund Terms.pdf", "title": "Chính Sách Hoàn Tiền Tiêu Chuẩn", "type": "legal", "year": 2025},
            "score": 0.91
        },
        {
            "content": "Người mua cần cung cấp bằng chứng hình ảnh hoặc video rõ nét chứng minh sản phẩm bị lỗi, bể vỡ, sai mẫu mã hoặc thiếu hàng làm căn cứ xử lý khiếu nại hoàn tiền.",
            "metadata": {"source": "Dispute Evidence Guide.md", "title": "Hướng Dẫn Cung Cấp Bằng Chứng", "type": "news", "year": 2026},
            "score": 0.83
        },
        {
            "content": "Phí vận chuyển trả hàng sẽ được Shopee hỗ trợ miễn phí hoàn toàn nếu người mua sử dụng đơn vị vận chuyển được Shopee chỉ định trong phiếu trả hàng.",
            "metadata": {"source": "Shipping Refund Guide.pdf", "title": "Quy Định Phí Vận Chuyển Hoàn Trả", "type": "legal", "year": 2026},
            "score": 0.72
        },
        {
            "content": "Sau khi người bán nhận lại hàng hoàn trả và xác nhận không khiếu nại thêm, Shopee sẽ xử lý hoàn lại tiền cho người mua vào Ví ShopeePay hoặc Tài khoản ngân hàng trong từ 2 đến 24 giờ làm việc.",
            "metadata": {"source": "Refund Processing Policy.md", "title": "Quy Trình Xử Lý Hoàn Tiền", "type": "news", "year": 2025},
            "score": 0.64
        }
    ],
    "bán": [
        {
            "content": "Người bán không được phép đăng tải các sản phẩm cấm theo pháp luật hiện hành và chính sách Shopee bao gồm: hàng giả, hàng nhái, các chất kích thích, vũ khí, động vật hoang dã và thuốc kê đơn.",
            "metadata": {"source": "Prohibited Products Policy.pdf", "title": "Chính Sách Sản Phẩm Cấm Đăng Bán", "type": "legal", "year": 2026},
            "score": 0.94
        },
        {
            "content": "Đối với người bán mới đăng tin sản phẩm đầu tiên, hệ thống yêu cầu bắt buộc xác thực số điện thoại di động và liên kết tài khoản ngân hàng chính chủ để kích hoạt tính năng bán hàng.",
            "metadata": {"source": "Seller Onboarding Guide.md", "title": "Hướng Dẫn Người Bán Mới", "type": "news", "year": 2025},
            "score": 0.87
        },
        {
            "content": "Tỉ lệ đơn hàng không thành công và tỉ lệ đơn hàng giao trễ vượt quá ngưỡng quy định sẽ dẫn đến việc shop bị tính điểm phạt Sao Quả Tạ và hạn chế tham gia các chương trình khuyến mãi.",
            "metadata": {"source": "Seller Penalty Points Policy.pdf", "title": "Chính Sách Điểm Phạt Người Bán", "type": "legal", "year": 2026},
            "score": 0.79
        },
        {
            "content": "Người bán có nghĩa vụ đảm bảo thông tin mô tả sản phẩm rõ ràng, chính xác, trung thực và hình ảnh đăng tải là hình ảnh thực tế để bảo vệ quyền lợi người tiêu dùng.",
            "metadata": {"source": "Seller Content Policy.md", "title": "Chính Sách Nội Dung Sản Phẩm", "type": "news", "year": 2025},
            "score": 0.71
        },
        {
            "content": "Shopee áp dụng biểu phí thanh toán và phí cố định tính trên mỗi đơn hàng giao thành công của người bán theo khung biểu phí được điều chỉnh hàng năm.",
            "metadata": {"source": "Seller Fees Policy.pdf", "title": "Quy Định Biểu Phí Người Bán", "type": "legal", "year": 2026},
            "score": 0.63
        }
    ]
}

def get_mock_response(query: str, use_reordering: bool = True):
    """Sinh câu trả lời mockup chân thực khi thiếu API Key hoặc chạy độc lập."""
    query_lower = query.lower()
    category = "thanh toán"
    if any(k in query_lower for k in ["đổi trả", "trả hàng", "hoàn tiền", "bằng chứng", "khiếu nại"]):
        category = "đổi trả"
    elif any(k in query_lower for k in ["bán", "người bán", "cấm", "sao quả tạ"]):
        category = "bán"

    chunks = MOCK_DOCUMENTS[category]
    
    if category == "thanh toán":
        answer = "Shopee hiện đang hỗ trợ nhiều phương thức thanh toán linh hoạt cho người mua. Bạn có thể chọn Thẻ tín dụng/ghi nợ, Ví điện tử ShopeePay, dịch vụ trả góp SPayLater, thanh toán khi nhận hàng COD, hoặc chuyển khoản ngân hàng [Chính Sách Thanh Toán Shopee, 2026]. \n\nĐể đổi phương thức thanh toán của đơn hàng đã đặt, bạn bắt buộc phải hủy đơn cũ và đặt đơn mới trước khi đơn hàng chuyển sang trạng thái chờ vận chuyển [Hướng Dẫn Quản Lý Đơn Hàng, 2025]. Lưu ý rằng hạn mức thanh toán qua ShopeePay được bảo mật ở mức tối đa là 50 triệu đồng mỗi ngày [Điều Khoản Bảo Mật ShopeePay, 2026]. Ngoài ra, nếu mua trả góp qua SPayLater, các kỳ hạn được áp dụng là 1, 3, 6 hoặc 12 tháng [Điều Khoản Dịch Vụ SPayLater, 2026]."
    elif category == "đổi trả":
        answer = "Thời hạn yêu cầu trả hàng và hoàn tiền trên hệ thống Shopee được phân chia theo phân khúc gian hàng. Đối với các đơn hàng mua từ gian hàng chính hãng Shopee Mall, bạn có thời hạn tối đa là 15 ngày để gửi yêu cầu [Chính Sách Hoàn Tiền Shopee Mall, 2026]. Đối với các shop thông thường không thuộc Mall, thời hạn khiếu nại đổi trả tiêu chuẩn là 3 ngày kể từ ngày đơn hàng hoàn thành [Chính Sách Hoàn Tiền Tiêu Chuẩn, 2025]. \n\nKhi thực hiện khiếu nại, bạn bắt buộc phải cung cấp đầy đủ bằng chứng bằng hình ảnh hoặc video cận cảnh sản phẩm bị lỗi [Hướng Dẫn Cung Cấp Bằng Chứng, 2026]. Phí ship gửi trả hàng sẽ hoàn toàn miễn phí nếu bạn sử dụng đơn vị vận chuyển được hệ thống chỉ định [Quy Định Phí Vận Chuyển Hoàn Trả, 2026]."
    else:
        answer = "Nhằm duy trì môi trường kinh doanh minh bạch, Shopee nghiêm cấm người bán đăng bán các mặt hàng vi phạm pháp luật hoặc quy định như hàng giả, vũ khí, thuốc kê đơn và chất kích thích [Chính Sách Sản Phẩm Cấm Đăng Bán, 2026]. \n\nĐối với các tài khoản người bán mới tạo, bạn cần hoàn tất việc xác thực số điện thoại và liên kết tài khoản ngân hàng chính chủ để bắt đầu bán hàng [Hướng Dẫn Người Bán Mới, 2025]. Bạn cũng có nghĩa vụ cung cấp mô tả sản phẩm trung thực để tránh tranh chấp [Chính Sách Nội Dung Sản Phẩm, 2025]. Nếu có nhiều đơn giao trễ hoặc bị hủy, shop của bạn có thể bị phạt điểm Sao Quả Tạ [Chính Sách Điểm Phạt Người Bán, 2026]."

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": "mock"
    }

# =============================================================================
# RAG EXECUTION PIPELINE
# =============================================================================

def run_rag_pipeline(query: str, top_k: int = 5, use_reordering: bool = True):
    """Chạy toàn bộ pipeline RAG: Retrieval -> (Reorder) -> Generation."""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    # Tự động dùng Mockup nếu không có API key hoặc thiếu thư viện backend
    if not HAS_BACKEND or not api_key:
        return get_mock_response(query, use_reordering)

    try:
        # Bước 1: Gọi hàm retrieve từ Task 9 (Semantic + Lexical + RRF Rerank)
        chunks = retrieve(query, top_k=top_k)
        
        if not chunks:
            return {
                "answer": "Không tìm thấy tài liệu phù hợp trong hệ thống dữ liệu.",
                "sources": [],
                "reordered_sources": [],
                "retrieval_source": "none"
            }

        # Bước 2: Reorder tài liệu để tránh "Lost in the Middle" (nếu được kích hoạt)
        if use_reordering:
            processed_chunks = reorder_for_llm(chunks)
        else:
            processed_chunks = chunks
            
        # Bước 3: Định dạng context
        context = format_context(processed_chunks)
        
        # Bước 4: Tạo Prompt và gửi đến LLM qua OpenRouter
        user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
        
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        answer = response.choices[0].message.content
        
        return {
            "answer": answer,
            "sources": chunks,  # Giữ danh sách gốc theo độ tương đồng để so sánh
            "reordered_sources": processed_chunks,
            "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
        }
    except Exception as e:
        # Dự phòng khẩn cấp nếu API gặp lỗi kết nối hoặc quota
        st.warning(f"Chuyển sang chế độ chạy thử nghiệm (Mockup Mode) do phát sinh lỗi: {e}")
        return get_mock_response(query, use_reordering)



# =============================================================================
# STREAMLIT UI DESIGN & INTERFACE
# =============================================================================

# Thiết lập giao diện hiện đại
st.set_page_config(
    page_title="RAG Task 10 - Reordering & Citation",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Chèn CSS tùy biến giúp ứng dụng đẹp mắt, chuyên nghiệp hơn
st.markdown("""
<style>
    /* Làm đẹp hộp chat */
    .stChatMessage {
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    /* Chỉnh nút bấm ở thanh bên */
    .stButton > button {
        border-radius: 8px;
        text-align: left;
        padding: 8px 12px;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #3b82f6;
        color: #3b82f6;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.1);
    }
    /* Expander tiêu đề */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# SIDEBAR CONTROL CENTER
# =============================================================================

with st.sidebar:
    st.title("📚 RAG Optimizer")
    st.caption("Trung tâm điều phối & Giám sát hiệu năng Mô hình Ngôn ngữ Lớn (LLM)")
    
    st.divider()
    
    # 1. Các câu hỏi gợi ý
    st.subheader("💡 Câu hỏi gợi ý nhanh")
    suggestions = [
        "Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?",
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "Làm sao để đổi phương thức thanh toán đơn hàng?",
        "Quy định về đăng bán sản phẩm cho người bán?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{hash(s)}"):
            st.session_state["pending_query"] = s
            st.rerun()

    st.divider()
    
    # 2. Các thiết lập tham số
    st.subheader("⚙️ Cấu hình Siêu Tham Số")
    top_k = st.slider("Số lượng tài liệu lấy ra (top_k)", min_value=3, max_value=10, value=5)
    
    # Switch bật tắt Document Reordering
    use_reordering = st.toggle(
        "Kích hoạt Document Reordering",
        value=True,
        help="Sắp xếp lại các đoạn văn theo cấu trúc xen kẽ (Đầu & Cuối prompt được ưu tiên) giúp LLM không bị bỏ sót các thông tin quan trọng nằm ở giữa."
    )
    
    st.divider()
    
    # Trạng thái backend để kiểm tra
    st.subheader("🌐 Trạng thái Hệ thống")
    if HAS_BACKEND:
        st.success("Backend: Đã kết nối cục bộ (Local Connection)")
    else:
        st.warning("Backend: Chế độ chạy thử (Sandbox Mode)")
        
    api_key_status = "Đã cấu hình" if (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")) else "Chưa cấu hình (Sử dụng Mockup)"
    st.info(f"API Key: {api_key_status}")

# =============================================================================
# MAIN WINDOW AREA
# =============================================================================

st.title("📚 E-commerce Support & RAG Verification Hub")
st.markdown("Hệ thống chatbot hỏi đáp chính sách e-commerce và hỗ trợ khách hàng tích hợp trích dẫn nguồn.")

# Khung hiển thị cuộc hội thoại
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Nếu là câu trả lời của trợ lý và có tài liệu nguồn kèm theo
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("📚 Chi Tiết Tài Liệu & Trích Dẫn"):
                for idx, src in enumerate(msg["sources"], 1):
                    meta = src.get("metadata", {})
                    title = meta.get("title") or meta.get("source") or "Tài liệu"
                    score = src.get("score", 0.0)
                    doc_type = meta.get("type", "unknown")
                    year = meta.get("year", "2026")
                    
                    st.markdown(f"**[{idx}] {title} ({year})**")
                    st.caption(f"Loại: `{doc_type}` | Score: `{score:.4f}` | File: `{meta.get('source', '')}`")
                    st.text_area("Nội dung trích đoạn:", value=src.get("content", ""), height=100, key=f"hist_{hash(title)}_{idx}")
                    st.divider()

# =============================================================================
# USER INPUT HANDLING
# =============================================================================

user_input = st.chat_input("Gõ câu hỏi của bạn về chính sách/hỗ trợ e-commerce...")
query = user_input or st.session_state.pending_query

if query:
    # Xóa trạng thái pending
    st.session_state.pending_query = None
    
    # Thêm và hiển thị câu hỏi của người dùng
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    # Tạo câu trả lời từ RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm nguồn tài liệu và tổng hợp câu trả lời có Citation..."):
            # Chạy truy vấn
            result = run_rag_pipeline(query, top_k=top_k, use_reordering=use_reordering)
            
            answer = result.get("answer", "Xin lỗi, hệ thống gặp lỗi khi truy xuất câu trả lời.")
            sources = result.get("sources", [])
            reordered_sources = result.get("reordered_sources", sources)
            
            # Hiển thị câu trả lời
            st.markdown(answer)
            
            # Hiển thị tài liệu nguồn
            if sources:
                with st.expander("📚 Chi Tiết Tài Liệu & Trích Dẫn", expanded=True):
                    for idx, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        title = meta.get("title") or meta.get("source") or "Tài liệu"
                        score = src.get("score", 0.0)
                        doc_type = meta.get("type", "unknown")
                        year = meta.get("year", "2026")
                        
                        st.markdown(f"**[{idx}] {title} ({year})**")
                        st.caption(f"Loại: `{doc_type}` | Score: `{score:.4f}` | File: `{meta.get('source', '')}`")
                        st.text_area("Nội dung trích đoạn:", value=src.get("content", ""), height=100, key=f"new_{hash(title)}_{idx}")
                        st.divider()
                            
    # Lưu vào lịch sử chat
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "reordered_sources": reordered_sources,
        "is_reordered": use_reordering
    })
    
    st.rerun()
