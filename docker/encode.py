from laser_encoders import initialize_encoder, initialize_tokenizer

tokenizer = initialize_tokenizer(lang="yor")
tokenized_sentence = tokenizer.tokenize("Eku aro")

encoder = initialize_encoder(lang="yor")
embeddings = encoder.encode_sentences([tokenized_sentence])

print("Embeddings Shape", embeddings.shape)