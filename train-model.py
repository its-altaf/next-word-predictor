"""
Trains the LSTM next-word model and saves it to disk so the Flask app
can load it instantly instead of retraining on every server start.

Run this once (or whenever you change the corpus):
    python train_model.py
"""

import os
import pickle
import json

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ---------------------------------------------------------------------------
# 1. Corpus - replace this with your own text data for real predictions
# ---------------------------------------------------------------------------
text = """
My name is Altaf Rahman, and I am an aspiring Artificial Intelligence engineer with a strong passion for technology, innovation, and continuous learning. I enjoy exploring new ideas and building intelligent systems that solve real-world problems. My primary interests are Machine Learning, Deep Learning, Computer Vision, Natural Language Processing, and Large Language Models. I believe that Artificial Intelligence has the potential to transform industries by automating repetitive tasks, improving decision-making, and providing intelligent solutions that help people in their daily lives.

My journey into Artificial Intelligence started with learning the fundamentals of programming using Python. As I became more comfortable with programming concepts, I started exploring data analysis, data visualization, statistics, and machine learning algorithms. I learned how data can be cleaned, transformed, and analyzed to discover meaningful patterns. This motivated me to continue my learning journey and dive deeper into deep learning and neural networks.

I enjoy learning modern AI technologies such as PyTorch, TensorFlow, FastAPI, Docker, LangChain, vector databases, Large Language Models, Retrieval-Augmented Generation, and cloud deployment tools. Every new technology teaches me something valuable about building scalable, efficient, and intelligent applications. Instead of only studying theory, I prefer implementing complete projects because practical experience helps me understand concepts more effectively.

One of my favorite areas is Natural Language Processing because it enables computers to understand, process, and generate human language. I find it fascinating that modern language models can answer questions, summarize long documents, translate languages, generate code, and assist users with complex tasks. Learning about transformers, attention mechanisms, embeddings, tokenization, and language modeling has helped me understand how these advanced AI systems work behind the scenes.

Currently, I am working on an AI-powered medical assistant project. The purpose of this application is to help users understand their medical reports, laboratory test results, prescriptions, discharge summaries, and treatment documents. Users can upload PDF files containing medical information, and the system extracts the text, processes it into smaller chunks, generates vector embeddings, and stores them in a vector database. When users ask questions, the system retrieves the most relevant information and uses a Large Language Model with Retrieval-Augmented Generation to generate accurate and context-aware responses. This project combines document processing, embeddings, vector search, prompt engineering, and conversational AI into a complete application.

Building this project has taught me many important concepts. I learned how document loaders work, how text splitting improves retrieval quality, how embeddings represent semantic meaning, how vector databases perform similarity searches, and how retrieval improves the accuracy of language models. I also learned how APIs connect the frontend with the backend, allowing users to interact with AI models through a simple and user-friendly interface.

Apart from Artificial Intelligence, I have experience in UI and UX design using Figma. I enjoy designing clean, responsive, and visually appealing interfaces for both mobile and web applications. I believe that a powerful AI system should also provide an excellent user experience because users should be able to interact with advanced technology without confusion. Good design makes software more accessible, intuitive, and enjoyable to use.

Previously, I designed travel booking applications that allowed users to search for flights, hotels, and transportation services. While designing these applications, I focused on creating intuitive navigation, simple booking workflows, and responsive layouts. I learned how color schemes, typography, spacing, icons, and user flows influence the overall experience of an application.

I also enjoy learning backend development because intelligent applications require reliable server-side infrastructure. I have been learning FastAPI to build RESTful APIs that communicate efficiently with frontend applications. FastAPI allows developers to create high-performance APIs with automatic documentation, data validation, and asynchronous programming support. I appreciate how modern backend frameworks simplify application development while maintaining excellent performance.

Docker is another technology that I enjoy learning because it makes applications portable and consistent across different environments. Containers eliminate many deployment issues by packaging the application, dependencies, and runtime into a single unit. This allows developers to deploy applications confidently without worrying about differences between development and production environments.

Machine Learning introduced me to supervised learning, unsupervised learning, feature engineering, model evaluation, and performance metrics. I learned about algorithms such as Linear Regression, Logistic Regression, Decision Trees, Random Forests, Support Vector Machines, Naive Bayes, K-Nearest Neighbors, and Gradient Boosting. Each algorithm has its own strengths and weaknesses, and selecting the appropriate algorithm depends on the characteristics of the dataset and the problem being solved.

Deep Learning expanded my understanding of neural networks, activation functions, optimization algorithms, backpropagation, convolutional neural networks, recurrent neural networks, LSTMs, GRUs, transformers, and attention mechanisms. I find it fascinating that deep learning models can recognize images, generate text, detect speech, translate languages, and perform many other intelligent tasks that were once considered impossible.

One of my goals is to become a Full Stack AI Engineer. I want to develop complete AI applications from frontend to backend, including data preprocessing, model training, API development, cloud deployment, monitoring, and user interface design. I believe that understanding every component of an AI system makes me a more effective engineer because I can identify problems and optimize the entire workflow instead of focusing on only one part.

Another area that interests me is Computer Vision. I want to build systems that can analyze medical images, detect diseases, recognize objects, understand scenes, and automate visual inspection tasks. I plan to study convolutional neural networks, image segmentation, object detection, image classification, pose estimation, and multimodal AI systems that combine images and text.

I enjoy reading research papers because they introduce me to the latest developments in Artificial Intelligence. Research papers often explain new architectures, optimization methods, datasets, evaluation techniques, and practical applications. Although some papers are challenging to understand at first, reading them regularly improves my knowledge and keeps me updated with recent advancements.

Problem-solving is one of my favorite activities. Whenever I encounter an error in my code, I try to understand its root cause instead of simply copying a solution. Debugging helps me improve my programming skills and teaches me how different software components interact. Every bug becomes an opportunity to learn something new.

I believe that consistency is more valuable than short bursts of motivation. Spending time learning every day allows me to build strong technical skills over time. Even small improvements accumulate into significant progress after weeks and months of dedicated practice. This mindset helps me stay focused on my long-term goals.

In the future, I hope to contribute to AI systems that improve healthcare, education, scientific research, and accessibility. I want to develop applications that help doctors analyze medical reports more efficiently, assist students in learning complex subjects, and make advanced technology accessible to people regardless of their technical background. My dream is to create intelligent systems that provide real value to society while remaining accurate, reliable, and easy to use.

Outside of programming, I enjoy exploring new technologies, reading technical articles, watching educational videos, participating in coding challenges, and experimenting with new ideas. Every project teaches me something different, whether it is software architecture, database design, machine learning optimization, user interface design, or cloud deployment. I believe that continuous learning is the foundation of success in technology because the field evolves rapidly and requires engineers to adapt to new tools and techniques.

As I continue my journey, I remain committed to improving my programming skills, strengthening my understanding of Artificial Intelligence, and building practical applications that solve meaningful problems. My goal is not only to become a skilled AI engineer but also to create innovative solutions that positively impact people's lives. Through dedication, curiosity, and continuous practice, I hope to contribute to the future of intelligent technology and inspire others to pursue learning and innovation.

"""


def build_and_train():
    # -----------------------------------------------------------------
    # 2. Tokenize
    # -----------------------------------------------------------------
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1
    print("Vocabulary size:", total_words)

    # -----------------------------------------------------------------
    # 3. Build input sequences (n-gram style)
    # -----------------------------------------------------------------
    input_sequences = []
    for line in text.split("\n"):
        if line.strip() == "":
            continue
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[: i + 1]
            input_sequences.append(n_gram_sequence)

    max_seq_len = max(len(x) for x in input_sequences)
    input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding="pre")

    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]
    y = to_categorical(y, num_classes=total_words)

    # -----------------------------------------------------------------
    # 4. Build the LSTM model
    # -----------------------------------------------------------------
    model = Sequential([
        Embedding(total_words, 100, input_length=max_seq_len - 1),
        LSTM(150, return_sequences=True),
        Dropout(0.2),
        LSTM(100),
        Dense(total_words, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    # -----------------------------------------------------------------
    # 5. Train
    # -----------------------------------------------------------------
    early_stop = EarlyStopping(patience=8, restore_best_weights=True, monitor="loss")
    model.fit(X, y, epochs=200, callbacks=[early_stop], verbose=1)

    # -----------------------------------------------------------------
    # 6. Save model, tokenizer, and config for the Flask app
    # -----------------------------------------------------------------
    os.makedirs("model", exist_ok=True)
    model.save("model/next_word_model.keras")

    with open("model/tokenizer.pickle", "wb") as f:
        pickle.dump(tokenizer, f)

    with open("model/config.json", "w") as f:
        json.dump({"max_seq_len": max_seq_len}, f)

    print("\nSaved model, tokenizer, and config to ./model/")


if __name__ == "__main__":
    build_and_train()