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
