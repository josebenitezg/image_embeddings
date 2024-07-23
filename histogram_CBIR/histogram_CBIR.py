import numpy as np

from datasets import load_dataset
from utils import process_fn, get_vector, cosine, build_histogram

def search(idx, vector_database,top_k=5):
    query_vector = vector_database[idx]
    distances = []
    for _, vector in enumerate(vector_database):
        distances.append(cosine(query_vector, vector))
    # get top k most similar images
    top_idx = np.argpartition(distances, -top_k)[-top_k:]
    return top_idx

if __name__ == "__main__":
    data = load_dataset('pinecone/image-set', split='train', revision='e7d39fc')
    image_dataset = [process_fn(sample) for sample in data]
    
    image_vector_database = []
    
    for image in image_dataset:
        image_vector_database.append(get_vector(image))
    
    search_results = search(0, image_vector_database)
    for result in search_results.tolist():
        print(result)
        build_histogram(image_dataset[result])
