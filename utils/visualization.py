from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

def mostrar_tsne(X, y, perplexity=10):
    
    target_names = ['0','1']
    # ---- t-SNE ----
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="random",
    )
    X_emb = tsne.fit_transform(X)

    # 4. Visualize the results
    plt.figure(figsize=(6, 5))
    colors = ['red', 'green', 'blue']

    for i, color, target_name in zip(range(len(target_names)), colors, target_names):
        plt.scatter(X_emb[y == i, 0], X_emb[y == i, 1], c=color, label=target_name)

    plt.legend()
    plt.title('t-SNE visualization of the Iris dataset')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    #plt.show()
    return plt