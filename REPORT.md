# báo cáo thực nghiệm memory systems cho ai agent - day 17

sinh viên: hoàng minh tuấn
môi trường test: conda env `VinAI` (python 3.11.14)

---

## 1. mấy số liệu đo đươc từ benchmark

em chạy file `src/benchmark.py` trong môi trường conda VinAI thì đo ra được hai bảng số liệu chi tiết như dưới đây. toàn bộ phân tích ở các mục sau em đều bốc thẳng từ các con số trong hai bảng này ra để đối chiếu chứ ko nói lý thuyết suông.

### bảng standard benchmark (chạy trên data/conversations.json - 10 hội thoại của user dungct)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline Agent | 4,504 | 29,291 | 0.0% | 0.10 | 0 | 0 |
| Advanced Agent | 3,865 | 35,890 | 100.0% | 1.00 | 393 | 2 |

### bảng long-context stress benchmark (chạy trên data/advanced_long_context.json - 16 turns dài dằng dặc của user dungct_stress)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline Agent | 740 | 28,724 | 0.0% | 0.10 | 0 | 0 |
| Advanced Agent | 1,797 | 11,548 | 100.0% | 1.00 | 317 | 28 |

---

## 2. phân tích 4 câu hỏi của bước 8 (lấy số liệu làm bằng chứng)

đúng theo hướng dẫn bước 8 trong guide, cứ mỗi ý em sẽ đưa số liệu ra trước rồi mới giải thích cơ chế code trong `src/` và chỉ ra giới hạn kỹ thuật đi kèm.

### vì sao advanced có recall ăn đứt baseline
nhìn vào cột `Cross-session recall` ở cả hai bảng thì thấy kết quả chênh lệch rât rõ: baseline ở cả 2 bảng đều tịt ngòi ở mức 0.0% (quality chỉ có 0.10), trong khi advanced thì đạt trọn vẹn 100.0% (quality 1.00). thêm vào đó là cột `Memory growth (bytes)` của baseline bằng 0 tròn trĩnh vì nó ko ghi file nào ra đĩa cả, còn advanced tăng lên 393 bytes ở standard và 317 bytes ở stress test.

cái này giải thích bằng code thì dễ hiểu thôi. trong `src/agent_baseline.py` con baseline chỉ lưu tạm tin nhắn vào ram trong biến `SessionState.messages` theo từng `thread_id`. lúc benchmark chuyển sang một thread mới toanh kiểu `f"{conv_id}-recall"` để hỏi lại thông tin cũ thì baseline coi như mất trí nhớ hoàn toàn, chả biết người dùng là ai nên recall bằng 0 là hiển nhiên. 

ngược lại bên `src/agent_advanced.py` đường đi của fact nó rât bài bản: 
đầu tiên hàm `extract_profile_updates()` trong `memory_store.py` bóc tách mấy thông tin cá nhân ổn định như tên tuổi, nơi ở, nghề nghiệp, sở thích; xong rồi gọi `UserProfileStore.save_facts()` ghi bền vững xuống đĩa thành file `state/profiles/<user>/User.md`; đến lúc sang thread mới hỏi recall thì hàm `_offline_response()` chỉ việc lôi file markdown này ra đọc lại là nhớ sạch sành sanh, kéo recall lên đúng 100%.

tuy nhiên giới hạn ở chổ này là nó phụ thuộc sống còn vào khâu bóc tách ban đầu. nếu lúc bóc tách regex hoặc model hiểu nhầm ý người dùng rồi ghi cứng vào file `User.md` thì sang các thread sau agent sẽ bị ảo giác vĩnh viễn (persistent error) vì cứ lôi fact sai trong file ra trả lời miết.

### vì sao advanced lại tốn hơn ở hội thoại ngắn
nhìn vào bảng standard benchmark sẽ thấy một điểm khá thú vị giữa hai cột token: cột `Agent tokens only` của advanced là 3,865 ít hơn baseline là 4,504 (do câu trả lời của advanced ngắn gọn đúng trọng tâm hơn), nhưng nhìn sang cột `Prompt tokens processed` thì baseline chỉ tốn 29,291 còn advanced lại ngốn tới 35,890 tokens, tức là advanced tốn hơn tầm 22.5% prompt context.

chổ này lúc đầu mới chạy xong nhìn bảng em cũng hơi giật mình, tự hỏi ủa sao advanced xịn hơn mà lại tốn prompt tokens hơn thế. ngồi ngẫm lại code mới thấy cực kì hợp lý:
- hội thoại trong bộ standard rât ngắn, mỗi session chỉ tầm chục câu ngắn ngủn, tổng lượng token của cả thread còn chưa chạm tới cái mốc `compact_threshold_tokens = 600`. bằng chứng là cột `Compactions` của advanced ở bảng standard chỉ nhảy có đúng 2 lần trên cả 10 hội thoại.
- trong khi đó, mỗi lượt gọi `_reply_offline()` của advanced thì hàm `_estimate_prompt_context_tokens()` lúc nào cũng phải vác thêm nguyên cả cái file `User.md` nhét vào prompt context gửi đi. thành ra lượt nào agent cũng phải cõng thêm một cái "ba lô" profile nặng tầm 80 đến 100 tokens. nhân lên cả chục phiên thì tổng prompt tokens bị đội lên thấy rõ, mà compact memory thì chưa đủ điều kiện kích hoạt để bù lại phần chi phí overhead mang theo file profile này.

giới hạn rút ra là: nếu làm chatbot cho mấy tác vụ ngắn kiểu người dùng vào hỏi 1-2 câu vu vơ rồi thoát, thì việc cứ nhét persistent profile vào mọi turn sẽ làm tốn tiền token đầu vào một cách lãng phí.

### vì sao compact memory lại thắng lớn ở hội thoại dài
sang đến cái bảng stress benchmark thì tình thế đảo chiều hoàn toàn. nhìn vào cột `Prompt tokens processed`: baseline ngốn tận 28,724 tokens, trong khi advanced chỉ tốn có 11,548 tokens, tức là advanced ép chi phí prompt context xuống tận 59.8%, tiết kiệm hơn một nửa tiền token đầu vào.

mà chổ này phải đặc biệt lưu ý một điểm quan trọng trong rubric: compact memory nó sinh ra là để tối ưu cột `Prompt tokens processed` này chứ ko phải để tối ưu cột `Agent tokens only`. bằng chứng rành rành là ở bảng stress, cột `Agent tokens only` của advanced là 1,797 thậm chí còn cao hơn baseline chỉ có 740 (vì advanced sinh câu trả lời đàng hoàng theo 3 bullet kèm ví dụ trade-off). cái giúp advanced ăn điểm chính là việc nó bóp nghẹt lượng prompt context truyền vào ở mỗi lượt.

nguyên nhân trong code là vì cái stress test này có 16 lượt dài dằng dặc về tin tức công nghệ, không gian, khí hậu... baseline ko có cơ chế nén nào cả, cứ lượt sau là bê nguyên xi tất cả các tin nhắn của các lượt trước nhét vào prompt làm token phình to theo cấp số cộng O(N^2). 
ngược lại bên advanced có `CompactMemoryManager` trong `src/memory_store.py` đứng canh: hễ thấy token trong thread vượt ngưỡng 600 tokens là nó cắt đống tin nhắn cũ (chỉ chừa lại 4 tin gần nhất) ném vào hàm `summarize_messages()` rồi dùng `merge_summaries()` gộp thành vài gạch đầu dòng tóm tắt ngắn gọn. cột `Compactions` nhảy lên tới 28 lần chứng tỏ nó nén liên tục qua các turn, nhờ thế mà prompt context lúc nào cũng bị chặn trần ko cho phình to.

tất nhiên cái gì cũng có giá của nó, giới hạn ở đây là nén summary bản chất là nén mất mát (lossy compression). nếu sau mười mấy lượt dài mà người dùng đột nhiên hỏi lại một chi tiết số liệu nhỏ nhặt trong quá khứ mà summary đã lỡ lược bỏ đi rồi thì agent sẽ chịu chết ko thể trả lời đươc.

### file memory tăng trưởng ra sao và mấy cái rủi ro đi kèm
nhìn cột `Memory growth (bytes)` thì thấy ở standard file tăng 393 bytes, còn ở stress tăng 317 bytes. file này do hàm `UserProfileStore.write_text()` ghi ra đĩa. trong phạm vi bài lab thì nhìn số byte này bé tí vài trăm byte thấy bình thường.

nhưng mà ngẫm kĩ rủi ro thục tế nếu đem hệ thống này chạy production lâu dài thì có hai vấn đề rất nguy hiểm:
thứ nhất là file bị phình to vô hạn theo thời gian (memory bloat): nếu người dùng chat cả năm trời mà cái gì agent cũng ghi thêm vào `User.md` thì file này sẽ dài cả nghìn dòng. mỗi lượt chat lại lôi cả nghìn dòng đó nhét vào prompt thì tiền token chịu sao thấu, chưa kể làm loãng ngữ cảnh và tràn context window.
thứ hai là rủi ro bị nhiễm độc dữ liệu sai (fact poisoning): ví dụ điển hình ở turn 15 của stress test, người dùng nói: *"có lúc mình đùa với đồng nghiệp rằng hay là chuyển sang product manager... nhưng đó chỉ là câu đùa"*, rồi *"hà nội chỉ là nơi mình vừa bay ra họp hai ngày chứ ko phải nơi ở"*. nếu hệ thống ngây thơ cứ thấy chức danh với địa danh là lưu vào profile thì coi như `User.md` bị dính fact rác, agent sẽ nhớ sai bét nhè ở các phiên sau.

---

## 3. chọn bonus và phân tích (đủ cả 3 vế theo rubric)

để giải quyết triệt để mấy cái rủi ro vừa nêu ở trên thì em chọn tập trung phân tích sâu vào 2 tính năng bonus chính mà em thấy cần thiết nhât dựa trên chính dữ liệu của bài lab: **Conflict Handling** và **Confidence Threshold** (kèm theo 2 cái phụ trợ là **Entity Extraction** và **Memory Decay**).

### A. Conflict Handling (xử lý xung đột khi có đính chính)
- **vấn đề thục tế gặp phải**: trong file `conversations.json` (ở conv-03 và conv-06) với cả turn 14 của stress test, người dùng thay đổi thông tin liên tục: lúc đầu bảo ở đà nẵng xong đính chính *"giờ mình đang ở huế chứ ko còn ở đà nẵng mỗi ngày nữa"*; lúc đầu làm backend engineer xong bảo *"mình ko còn làm backend nữa, giờ chuyển sang MLOps engineer"*; rồi trong stress test lại từ huế về đà nẵng làm việc. nếu hệ thống ko biết xử lý xung đột mà cứ lưu dồn thì trong `User.md` sẽ tồn tại song song cả 2 thông tin đá nhau chan chát. sang thread mới hỏi lại là agent bị lú ngay (hallucination), ko biết chọn thông tin nào.
- **cải thiện số liệu thế nào**: trong `src/memory_store.py`, hàm `extract_profile_updates()` của em có logic bắt các cụm từ đính chính (`đính chính`, `ko còn làm`, `chuyển sang`, `cập nhật từ... sang...`). hễ có fact mới được xác nhận là nó đè bẹp fact cũ luôn, đồng thời ghi một dòng log `[CONFLICT RESOLVED] Location: updated Huế -> Đà Nẵng (confidence: 0.95)` ở cuối file. nhờ cơ chế này mà ở mấy câu hỏi bẫy hóc búa như *"Nếu ai đó nhắc Huế, Hà Nội hay product manager, đâu mới là nghề nghiệp và nơi ở hiện tại của mình?"*, Advanced Agent vẫn trả lời đúng phóc, giúp `Cross-session recall` đạt trọn vẹn 100.0% và file memory giữ được mức 317 bytes ko bị rác.
- **rủi ro sinh ra cho hệ thống**: cái giá phải trả của việc ghi đè là nguy cơ **ghi đè nhầm (false overwrite)**. lỡ người dùng chỉ đang lấy một ví dụ giả định hoặc kể chuyện về người khác (kiểu "đồng nghiệp mình mới chuyển vào đà nẵng") mà bộ bóc tách bắt nhầm chủ ngữ thì fact đúng của người dùng bị xóa sổ luôn, ko thể rollback lại được nếu ko có hệ thống versioning lịch sử đi kèm.

### B. Confidence Threshold (đặt ngưỡng tin cậy lọc câu đùa và nhiễu)
- **vấn đề thục tế gặp phải**: ở turn 15 của stress test có quả bẫy cực kì hiểm: người dùng bảo đùa muốn làm "product manager", rồi nhắc "hà nội" chỉ là nơi ra họp công tác 2 ngày. ngoài ra người dùng thỉnh thoảng hay hỏi vặn lại kiểu *"Bạn có biết DũngCT ko?"*, *"Hiện tại mình đang ở đâu?"*. nếu ko có ngưỡng tin cậy thì bot sẽ lưu luôn nghề là PM, nơi ở là Hà Nội, hoặc tệ hơn là coi câu hỏi của người dùng thành fact mới.
- **cải thiện số liệu thế nào**: em code thêm bộ lọc độ tin cậy trong `extract_profile_updates()`: mấy câu hỏi tu từ ko có từ khóa tự giới thiệu thì gán confidence = 0 bỏ qua luôn; mấy câu chứa từ khóa đùa hoặc phủ định (`câu đùa`, `chỉ là nơi bay ra họp`, `ko phải nơi ở`) thì lập tức chặn ko cho cập nhật. nhờ thế mà nghề MLOps engineer và nơi ở Đà Nẵng được bảo toàn nguyên vẹn, giúp ăn trọn điểm ở câu hỏi recall số 2 của stress test.
- **rủi ro sinh ra cho hệ thống**: code phức tạp hơn hẳn và có rủi ro bị **lọc sót fact thật (false negative)**. nếu người dùng nói chuyện theo kiểu nói giảm nói tránh hoặc dùng văn phong ẩn dụ mà bộ lọc tưởng là nói đùa rồi bỏ qua thì bot sẽ ko chịu cập nhật thông tin mới.

### C. Hai tính năng phụ trợ: Entity Extraction có cấu trúc & Memory Decay
- **Entity Extraction**: thay vì lưu văn bản tự do lộn xộn, em ép về 8 trường cố định (Name, Location, Profession, Style, Drink, Food, Pet, Interests). việc này giúp file `User.md` nhìn rât ngăn nắp, lúc cần đọc fact chỉ mất $O(1)$ là lôi ra được, prompt gửi đi cũng sạch sẽ hơn nhiều.
- **Memory Decay (sliding window log)**: cái phần log xung đột ở cuối `User.md` em chặn trần chỉ cho giữ tối đa 8 dòng log mới nhât thôi. fact cũ mèm sẽ bị dọn bớt đi để giữ kích thước file ko bị phình to vô tội vạ sau thời gian dài sử dụng.

---

## 4. xâu chuỗi 5 mắt xích theo đúng yêu cầu của rubric

nói tóm lại một câu, cả bài lab này có thể thấy rõ 5 mắt xích ăn khớp với nhau qua từng con số đo đạc cụ thể:

1. **baseline ko nhớ dài hạn**: minh chứng bằng con số `0.0%` recall và `0 bytes` memory vì nó chỉ lưu ram trong session, sang thread mới là quên sạch sành sanh.
2. **advanced thêm User.md giúp recall tăng vọt**: minh chứng bằng con số `100.0%` recall và `393 bytes` memory ở bảng standard, nhờ fact đi từ `extract_profile_updates()` vào `User.md` rồi được `_offline_response()` lôi ra trả lời.
3. **hội thoại dài làm prompt token của baseline nổ tung**: minh chứng bằng con số `28,724` prompt tokens ở bảng stress do context của baseline phình to theo cấp số cộng O(N^2) qua 16 lượt dài dằng dặc.
4. **compact memory kéo chi phí ngữ cảnh xuống cực mạnh**: minh chứng bằng con số `11,548` prompt tokens ở advanced (tiết kiệm gần 60% so với baseline) cùng `28` lần compactions tự động, khẳng định compact sinh ra để tối ưu prompt context chứ ko phải agent output tokens.
5. **hệ thống mạnh hơn nhưng phức tạp hơn và cần guardrail**: minh chứng bằng mấy cái ca đính chính huế/đà nẵng và câu đùa PM, bắt buộc phải có `Conflict handling` và `Confidence threshold` để giữ cho bộ nhớ ko bị ảo giác.

---

## 5. lệnh chạy kiểm thử lại trên máy

em đã test cẩn thận trên terminal powershell với môi trường conda VinAI:

```powershell
# chạy 9 bài unit test (pass trọn vẹn 9/9 cả test lõi lẫn test bonus trong 0.12s)
& "C:\Users\ironh\anaconda3\envs\VinAI\python.exe" -m pytest src/test_agents.py -v

# chạy benchmark xuất 2 bảng và tự động ghi báo cáo
& "C:\Users\ironh\anaconda3\envs\VinAI\python.exe" src/benchmark.py
```

toàn bộ code và logic đã hoàn thiện chuẩn chỉnh, chạy mượt mà và ko gặp lỗi nào.
