# Phase 2, Track 3, Day 17: Memory Systems for AI Agent

Trong Day 17 này, các bạn sẽ tập trung vào một câu hỏi rất thực tế: làm sao để AI agent **không chỉ trả lời tốt trong một lượt chat**, mà còn **nhớ đúng thông tin quan trọng qua nhiều phiên làm việc** mà vẫn kiểm soát được chi phí token.

Trong bài lab này, các bạn sẽ xây dựng và so sánh hai agent:

- `Baseline Agent`: chỉ có short-term memory trong cùng một thread
- `Advanced Agent`: có short-term memory, `User.md` bền vững, và compact memory để nén hội thoại dài

Mục tiêu cuối cùng không phải chỉ là “agent nhớ nhiều hơn”, mà là hiểu rõ trade-off giữa:

- độ nhớ dài hạn
- chất lượng phản hồi
- chi phí token
- độ phức tạp của hệ thống memory

## Các bạn sẽ làm gì trong track này?

Sau khi hoàn thành, các bạn cần có khả năng:

- phân biệt `short-term memory`, `persistent memory`, và `compact memory`
- xây dựng agent baseline và advanced trên cùng một benchmark
- lưu hồ sơ người dùng bằng `User.md`
- kích hoạt compact memory khi hội thoại dài vượt ngưỡng
- benchmark hai agent bằng cùng một bộ dữ liệu tiếng Việt
- đọc kết quả benchmark theo các chỉ số recall, token, memory growth, chất lượng phản hồi

## Cấu trúc codebase

```
.
├── README.md        # giới thiệu track (file này)
├── Guide.md         # hướng dẫn từng bước
├── Rubric.md        # tiêu chí chấm điểm
├── data/            # dữ liệu benchmark dùng chung
│   ├── conversations.json
│   └── advanced_long_context.json
└── src/             # bản scaffold dành cho sinh viên (pseudocode + TODO)
    ├── model_provider.py
    ├── config.py
    ├── memory_store.py
    ├── agent_baseline.py
    ├── agent_advanced.py
    ├── benchmark.py
    └── test_agents.py
```

Khi chạy, agent sẽ ghi trạng thái (ví dụ `state/profiles/<user>/User.md`) vào thư mục `state/`. Thư mục này đã nằm trong `.gitignore`.

### Vai trò từng file trong `src/`

Các file được liệt kê theo thứ tự nên triển khai:

| File | Vai trò | Thành phần chính |
|---|---|---|
| `model_provider.py` | Khởi tạo chat model cho từng provider | `ProviderConfig`, `normalize_provider()`, `build_chat_model()` |
| `config.py` | Cấu hình chung của lab | `LabConfig` (đường dẫn, ngưỡng compact, model chính + judge), `load_config()` |
| `memory_store.py` | Lõi memory layer | `estimate_tokens()`, `UserProfileStore` (read/write/edit `User.md`), `extract_profile_updates()`, `summarize_messages()`, `CompactMemoryManager` |
| `agent_baseline.py` | Agent A: chỉ nhớ trong cùng thread | `BaselineAgent.reply()`, `token_usage()`, `prompt_token_usage()` |
| `agent_advanced.py` | Agent B: short-term + `User.md` + compact | `AdvancedAgent.reply()`, `_reply_offline()`, `_estimate_prompt_context_tokens()`, `_offline_response()` |
| `benchmark.py` | So sánh hai agent trên hai bộ dữ liệu | `run_agent_benchmark()`, `recall_points()`, `heuristic_quality()`, `format_rows()` |
| `test_agents.py` | Kiểm chứng hành vi memory | test `User.md`, compact trigger, cross-session recall, giảm prompt load |

### Luồng xử lý một lượt của Advanced Agent

```
message người dùng
  → extract_profile_updates()      # trích fact ổn định: tên, nơi ở, nghề, style...
  → ghi vào User.md                # persistent memory
  → CompactMemoryManager.append()  # short-term memory, tự compact khi vượt ngưỡng
  → prompt = User.md + summary + recent messages
  → sinh câu trả lời → cập nhật bộ đếm token
```

Baseline Agent chỉ giữ danh sách message theo `thread_id`. Sang thread mới, nó **phải quên** toàn bộ fact cũ.

Cả hai agent nên có **chế độ offline** cho ra kết quả lặp lại được, để benchmark và test chạy được mà không cần API key. Chế độ live (LangChain/LangGraph) là phần mở rộng.

## Dữ liệu benchmark

| File | Nội dung | Mục tiêu |
|---|---|---|
| `data/conversations.json` | 10 hội thoại khoảng 10 lượt, user `dungct`, kèm `recall_questions` | Standard benchmark: đo recall qua nhiều phiên bình thường |
| `data/advanced_long_context.json` | 1 hội thoại 16 lượt rất dài, user `dungct_stress` | Long-context stress benchmark: ép compact xảy ra nhiều lần |

Mỗi hội thoại có dạng:

```json
{
  "id": "conv-01",
  "user_id": "dungct",
  "turns": ["...", "..."],
  "recall_questions": [
    { "question": "...", "expected_contains": ["DũngCT", "cà phê sữa đá"] }
  ]
}
```

`recall_questions` được hỏi ở **thread mới**. Điểm recall dựa trên số chuỗi trong `expected_contains` xuất hiện trong câu trả lời.

Dữ liệu cố tình chứa các tình huống khó:

- **correction**: nơi ở đổi giữa Đà Nẵng và Huế, agent phải giữ fact mới nhất
- **nhiễu**: "Hà Nội" chỉ là nơi đi họp, "product manager" chỉ là câu đùa
- **ngữ cảnh dài**: nhiều đoạn tin tức dài trong stress test để làm lộ chi phí prompt của baseline

## Provider hỗ trợ

Trong bản solved lab, runtime hỗ trợ các provider sau:

- `openai`
- `custom` (OpenAI-compatible base URL)
- `gemini`
- `anthropic`
- `ollama`
- `openrouter`

Điều này quan trọng vì memory system không nên bị khóa vào một provider duy nhất.

## Chỉ số benchmark cần hiểu

Khi hoàn thiện bài, benchmark nên cho các cột sau:

- `Agent tokens only`: token sinh ra trực tiếp trong hội thoại của agent
- `Prompt tokens processed`: lượng ngữ cảnh agent phải kéo theo qua các lượt
- `Cross-session recall`: khả năng nhớ facts qua thread hoặc session mới
- `Response quality`: chất lượng phản hồi
- `Memory growth (bytes)`: tốc độ phình của file memory
- `Compactions`: số lần compact memory đã nén lịch sử cũ

Điểm quan trọng nhất của track này là:

- ở hội thoại ngắn, `Advanced` có thể tốn hơn `Baseline` về token usage
- ở hội thoại rất dài, compact memory nên giúp `Advanced` xử lý ngữ cảnh hiệu quả hơn đáng kể + tiết kiệm usage.

## Setup môi trường

Các bạn cần chuẩn bị môi trường Python `>= 3.11` và cài các package cần thiết cho LangChain, LangGraph, provider SDK, `python-dotenv`, `tabulate`, và `pytest`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest
```

Nếu muốn chạy chế độ live với LLM thật, hãy tạo file `.env` ở root repo (đã nằm trong `.gitignore`). Tên biến môi trường do các bạn quyết định khi viết `load_config()`. Ví dụ:

```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=...
```

## Chạy benchmark và test

Sau khi hoàn thiện `src/`, chạy từ root repo:

```bash
python src/benchmark.py
```

```bash
pytest src/test_agents.py -v
```

Benchmark cần in ra hai bảng: **Standard Benchmark** và **Long-Context Stress Benchmark**. Mỗi bảng so sánh Baseline với Advanced theo đủ 6 cột trong phần "Chỉ số benchmark cần hiểu".

## Cách dùng repo này

Nếu các bạn là sinh viên:

- làm bài trong `src/`
- dùng `data/` làm benchmark input

Nếu các bạn là giảng viên hoặc reviewer:

- dùng `src/` để đánh giá scaffold giao cho sinh viên và kết quả hoàn thiện cuối cùng

## Tài liệu nên đọc tiếp

- `Guide.md`: hướng dẫn từng bước để hoàn thành lab
- `Rubric.md`: tiêu chí chấm điểm và bonus

Track này được thiết kế để các bạn không chỉ “dùng agent”, mà còn bắt đầu nghĩ như một người thiết kế **memory system** cho agent production.

---

---

## tóm tắt báo cáo thực nghiệm & phân tích kỹ thuật (day 17)

toàn bộ báo cáo chi tiết xem tại file [REPORT.md](file:///c:/Users/ironh/Downloads/VinUni/Lab/day17-HoangMinhTuan-2A202602758-MemorySystems4Agent/REPORT.md).

### 1. bảng kết quả benchmark thực nghiệm

môi trường chạy: conda env `VinAI` (python 3.11.14)

#### standard benchmark (data/conversations.json - 10 hội thoại của user dungct)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline Agent | 4,504 | 29,291 | 0.0% | 0.10 | 0 | 0 |
| Advanced Agent | 3,865 | 35,890 | 100.0% | 1.00 | 393 | 2 |

#### long-context stress benchmark (data/advanced_long_context.json - 16 turns dài dằng dặc của user dungct_stress)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline Agent | 740 | 28,724 | 0.0% | 0.10 | 0 | 0 |
| Advanced Agent | 1,797 | 11,548 | 100.0% | 1.00 | 317 | 28 |

---

### 2. phân tích nhanh kết quả đo đươc (trả lời 4 câu hỏi bước 8)

em bốc thẳng số liệu thực tế đo được từ hai cái bảng ở trên ra giải thích:

1. **vì sao advanced có recall ăn đứt baseline**:
   nhìn cột `Cross-session recall`, baseline ở cả 2 bảng đều tịt ngòi 0.0% (quality có 0.10), còn advanced thì ăn trọn 100.0% (quality 1.00). cột `Memory growth (bytes)` của baseline bằng 0 vì chẳng ghi file nào, còn advanced tăng 393 bytes ở standard và 317 bytes ở stress.
   trong code `src/agent_baseline.py`, baseline chỉ lưu tạm trong ram (`SessionState.messages`), sang thread mới hỏi recall là mù tịt hoàn toàn. còn advanced trong `src/agent_advanced.py` đi theo đường: `extract_profile_updates()` bóc tách fact -> `UserProfileStore.save_facts()` ghi bền vững vào `User.md` -> `_offline_response()` đọc lại file trong thread mới để trả lời đúng phóc 100%. giới hạn là nếu khâu bóc tách ban đầu bị sai thì fact sai sẽ bị lưu vĩnh viễn (persistent error).

2. **vì sao advanced lại tốn hơn ở hội thoại ngắn**:
   ở bảng standard, cột `Agent tokens only` của baseline là 4,504 và advanced là 3,865. nhưng ở cột `Prompt tokens processed`, baseline chỉ tốn 29,291 trong khi advanced tốn tới 35,890 tokens (advanced tốn hơn tầm 22.5% prompt context).
   nguyên nhân vì hội thoại trong standard rât ngắn, chưa chạm ngưỡng `compact_threshold_tokens = 600` nên cột `Compactions` chỉ nhảy có 2 lần. mà mỗi lượt gọi `_reply_offline()`, hàm `_estimate_prompt_context_tokens()` của advanced lúc nào cũng phải vác thêm nguyên cả cái file `User.md` nhét vào prompt context. thành ra hội thoại ngắn chưa đủ dài để compact bù lại phần chi phí overhead mang theo file profile này.

3. **vì sao compact có lợi thế ở hội thoại dài**:
   ở bảng stress, cột `Prompt tokens processed` của baseline lên tới 28,724 tokens, trong khi advanced chỉ tốn 11,548 tokens (tiết kiệm gần 60%). đáng chú ý là cột `Agent tokens only` của advanced (1,797) lại cao hơn baseline (740), chứng minh rõ ràng compact **chủ yếu tối ưu cột `Prompt tokens processed`** chứ ko phải agent tokens sinh ra.
   stress test có 16 lượt dài dằng dặc làm context của baseline phình to theo cấp số cộng O(N^2). ngược lại, lớp `CompactMemoryManager` của advanced tự động kích hoạt nén liên tục (cột `Compactions = 28`), cắt tin cũ đưa vào `summarize_messages()` và `merge_summaries()` để giữ prompt context luôn gọn gàng trong giới hạn. giới hạn là nén summary bản chất là nén mất mát (lossy compression), chấp nhận lược bỏ chi tiết vụn vặt trong quá khứ.

4. **file memory tăng trưởng ra sao và rủi ro gì**:
   cột `Memory growth (bytes)` tăng 393 bytes (standard) và 317 bytes (stress) do `UserProfileStore.write_text()` ghi ra đĩa. rủi ro thục tế quan sát được là: (1) nếu ko kiểm soát, file sẽ phình to liên tục theo thời gian làm đội chi phí prompt context (memory bloat); (2) nguy cơ lưu nhầm thông tin nhiễu hoặc câu nói đùa của người dùng (ví dụ nói đùa muốn làm product manager ở turn 15).

---

### 3. chọn tính năng bonus (rubric 90-100 điểm)

em chọn và phân tích sâu 2 hướng bonus chính gắn liền với điểm yếu thực tế trong dữ liệu:

- **Conflict Handling (xử lý xung đột khi có đính chính)**:
  - *vấn đề*: user đổi nơi ở từ đà nẵng sang huế, đổi nghề từ backend sang mlops (conv-03, conv-06, stress turn 14). nếu ko xử lý thì 2 fact mâu thuẫn cùng tồn tại trong User.md gây ảo giác.
  - *cải thiện*: tự động ghi đè fact cũ, ghi log `[CONFLICT RESOLVED]`, giúp `Cross-session recall` đạt tuyệt đối 100.0%.
  - *rủi ro*: nguy cơ ghi đè nhầm (false overwrite) nếu user chỉ đang lấy ví dụ giả định mà parser hiểu nhầm là đính chính.
- **Confidence Threshold (lọc nhiễu và câu đùa)**:
  - *vấn đề*: user đùa chuyển sang "product manager" và nói "hà nội chỉ là nơi đi họp 2 ngày".
  - *cải thiện*: lọc bỏ câu đùa và câu hỏi tu từ, bảo toàn đúng nghề MLOps và nơi ở Đà Nẵng.
  - *rủi ro*: ngưỡng lọc quá khắt khe có thể bỏ sót thông tin thực sự nếu user nói gián tiếp (false negative).
- bổ sung thêm **Entity Extraction** (chuẩn hóa schema $O(1)$) và **Memory Decay** (giữ sliding window tối đa 8 dòng log) để chặn đứng nguy cơ phình to file memory.

---

### 4. lệnh chạy kiểm thử

```powershell
# chạy toàn bộ 9 unit tests trong env VinAI (pass 100% trong 0.12s)
& "C:\Users\ironh\anaconda3\envs\VinAI\python.exe" -m pytest src/test_agents.py -v

# chạy benchmark xuất 2 bảng và tạo file report
& "C:\Users\ironh\anaconda3\envs\VinAI\python.exe" src/benchmark.py
```




