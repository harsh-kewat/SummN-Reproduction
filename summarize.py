# from transformers import pipeline

# # Load BART summarizer model
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# text = """
# The project manager opened the meeting and recapped the decisions made in the previous meeting. The marketing expert discussed his personal
# preferences for the design of the remote and presented the results of trend-watching reports , which indicated that there is a need for products
# which are fancy , innovative , easy to use , in dark colors , in recognizable shapes , and in a familiar material like wood. The user interface
# designer discussed the option to include speech recognition and which functions to include on the remote. The industrial designer discussed
# which options he preferred for the remote in terms of energy sources , casing , case supplements , buttons , and chips. The team then discussed
# and made decisions regarding energy sources , speech recognition , LCD screens , chips , case materials and colors, case shape and orientation ,
# and button orientation.· · · The case covers will be available in wood or plastic. The case will be single curved. Whether to use kinetic energy or
# a conventional battery with a docking station which recharges the remote. Whether to implement an LCD screen on the remote. Choosing
# between an LCD screen or speech recognition. Using wood for the case.
# """

# summary = summarizer(text, max_length=60, min_length=20, do_sample=False)
# print("\n🟢 Summary:\n", summary[0]['summary_text'])
# from transformers import pipeline

# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# text = open("icsi_sample.txt").read()
# chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

# for i, chunk in enumerate(chunks):
#     print(f"\n---- Summary Chunk {i+1} ----")
#     summary = summarizer(chunk, max_length=100, min_length=40, do_sample=False)
#     print(summary[0]["summary_text"])
# from transformers import pipeline

# # Load summarizer
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# # Input text (you can replace this with your meeting transcript)
# text = """
# The project manager opens the meeting by recapping the events of the previous meeting...
# (put full meeting text here)
# """

# # Split text into smaller chunks (~800 tokens each)
# def chunk_text(text, max_chunk_size=1000):
#     sentences = text.split(". ")
#     chunks, current_chunk = [], ""
#     for sentence in sentences:
#         if len(current_chunk) + len(sentence) < max_chunk_size:
#             current_chunk += sentence + ". "
#         else:
#             chunks.append(current_chunk.strip())
#             current_chunk = sentence + ". "
#     if current_chunk:
#         chunks.append(current_chunk.strip())
#     return chunks

# chunks = chunk_text(text)
# final_summary = ""

# print(f"Total Chunks: {len(chunks)}\n")

# for i, chunk in enumerate(chunks, start=1):
#     input_len = len(chunk.split())
#     max_len = max(14, int(input_len * 0.4))  # output ≈ 40% of input length
#     min_len = int(max_len * 0.5)

#     summary = summarizer(
#         chunk,
#         max_length=max_len,
#         min_length=min_len,
#         do_sample=False
#     )[0]['summary_text']

#     print(f"---- Summary Chunk {i} ----")
#     print(summary, "\n")
#     final_summary += summary + " "

# print("\n🟢 FINAL SUMMARY:\n", final_summary.strip())
from transformers import pipeline

# Load BART-large summarizer (powerful summarization model)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Your long meeting text
text = """
The project manager opens the meeting by recapping the events of the previous meeting. The marketing expert presents the results of market
research , which shows that users want a fancy-looking remote control that is easy to use and has a fancy look and feel. The user interface
designer presents the user interface concept for the remote , which is based on the idea that a remote should be simple and user-friendly.
The industrial designer presents about the internal components of a remote control. The group discusses using kinetic energy to power the
device , using a simple battery for the LCD screen , and using an advanced chip for the advanced chip. The project manager closes the meeting
, telling the team members what their tasks will be for the next meeting. · · · The Marketing Expert will research how to produce a remote that
is technologically innovative. The User Interface Designer will look at how to make a remote out of wood or plastic with either a wooden
or plastic cover. The Group will not work with teletext. There was a lack of information on the cost of components &materials.
"""

# Function to split large text into overlapping chunks
def chunk_text(text, max_chunk_size=900):
    sentences = text.split(". ")
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            # small overlap for context retention
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

chunks = chunk_text(text)
final_summary = ""

print(f"🧩 Total Chunks: {len(chunks)}\n")

# Summarize each chunk with tuned parameters
for i, chunk in enumerate(chunks, start=1):
    input_len = len(chunk.split())
    # dynamic max/min lengths to keep output rich
    max_len = min(180, int(input_len * 0.9))
    min_len = int(max_len * 0.6)

    summary = summarizer(
        chunk,
        max_length=max_len,
        min_length=min_len,
        do_sample=True,        # enables richer phrasing
        temperature=0.7,       # adds creativity but keeps focus
        repetition_penalty=1.2 # avoids repeated phrases
    )[0]['summary_text']

    print(f"---- Summary Chunk {i} ----")
    print(summary, "\n")
    final_summary += summary + " "

# Final combined summary
print("\n🟢 FINAL SUMMARY:\n", final_summary.strip())
