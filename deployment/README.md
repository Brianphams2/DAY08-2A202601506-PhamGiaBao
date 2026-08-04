# Deploy Streamlit + Vercel

Không đưa trực tiếp app Streamlit/BGE-M3/ChromaDB vào Vercel Function. Vercel
được dùng làm landing page; chatbot chạy trên Streamlit Community Cloud và được
nhúng bằng `?embed=true`.

## 1. Deploy chatbot lên Streamlit Community Cloud

1. Push repository lên GitHub, mở Streamlit Community Cloud và chọn **Create app**.
2. Chọn repository, branch cần deploy và entrypoint `app.py`.
3. Trong **Advanced settings > Secrets**, dán nội dung theo
   `.streamlit/secrets.toml.example` và thay bằng key thật.
4. Deploy và chờ BGE-M3 được tải ở lần cold start đầu tiên.
5. Kiểm tra URL dạng `https://YOUR-APP.streamlit.app` và thử ít nhất một câu hỏi.

## 2. Deploy landing page lên Vercel

1. Trong `deployment/vercel-landing/index.html`, thay `YOUR-APP` bằng subdomain
   Streamlit vừa nhận được.
2. Trên Vercel, import cùng GitHub repository và đặt **Root Directory** là
   `deployment/vercel-landing`.
3. Giữ Framework Preset là **Other**, không cần Build Command hoặc Output Directory.
4. Deploy, mở URL Vercel và xác nhận chatbot hiển thị trong iframe.

Nếu Streamlit từ chối nhúng, kiểm tra URL iframe có hậu tố `/?embed=true`. Không
commit `.env` hoặc `.streamlit/secrets.toml`; cả hai đã được `.gitignore` bảo vệ.

Tài liệu chính thức: [Deploy Streamlit app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy),
[embed Streamlit app](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app/embed-your-app)
và [Vercel runtimes](https://vercel.com/docs/functions/runtimes).
