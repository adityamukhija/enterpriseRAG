# test_chunking.py (run this to verify)
from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import RecursiveChunker

# Create a sample doc
with open("documents/sample.txt", "w") as f:
    f.write("""
Hotel Grand Dubai - Property Description

The Hotel Grand Dubai is a luxury 5-star property located in the heart of 
Dubai Marina. The hotel features 350 elegantly appointed rooms and suites, 
each offering panoramic views of the Arabian Gulf or the stunning city skyline.

Amenities and Facilities

Guests can enjoy a world-class spa, infinity pool overlooking the marina, 
state-of-the-art fitness center, and five dining restaurants including an 
award-winning rooftop establishment. The hotel also features a private beach 
club and concierge services available 24/7.

Cancellation Policy

Free cancellation is available up to 48 hours before the scheduled check-in 
date. Cancellations made within 48 hours will incur a charge equal to one 
night's stay. No-shows will be charged the full booking amount. Group bookings 
of 5 or more rooms require 14 days advance notice for cancellation.

Check-in and Check-out

Standard check-in time is 3:00 PM and check-out is 12:00 PM. Early check-in 
and late check-out are subject to availability and may incur additional charges. 
Express check-in is available for loyalty program members.
""")

loader = DocumentLoader()
chunker = RecursiveChunker(chunk_size=500, chunk_overlap=100)

pages = loader.load("documents/sample.txt")
chunks = chunker.chunk(pages)

print(f"Document split into {len(chunks)} chunks:\n")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i} ({len(chunk['content'])} chars) ---")
    print(chunk["content"][:200] + "...")
    print()