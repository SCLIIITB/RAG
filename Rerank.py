from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# model = SentenceTransformer(model_name)


def getRerankDocs(documents, query):
    pairs = []
    for doc in documents:
        pairs.append([query, doc])
    scores = cross_encoder.predict(pairs)
    print("Cross Encoder Scores: ", scores)
    scored_docs = zip(scores, documents)
    reranked_document_cross_encoder = sorted(scored_docs, reverse=True)
    reranked_document = [item for score,item in reranked_document_cross_encoder]
    return (reranked_document)