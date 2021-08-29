from functions import find_vector, n_closest_words
import numpy as np
import wikipedia2vec

# %%

def main():
    king = find_vector("מלך")
    man = find_vector("גבר")
    woman = find_vector("אישה")
    queen_ = king - man + woman
    queen = find_vector("מלכה")
    norm = np.linalg.norm(queen/np.linalg.norm(queen) - queen_/np.linalg.norm(queen_))
    angle = (queen_ @ queen.T) / (np.linalg.norm(queen)*np.linalg.norm(queen_))
    print("Norm is: {n}, angle is: {a}".format(n=norm, a=angle))
    #queen_words = n_closest_words(queen, n=5)
    #for i, word in enumerate(queen_words):
     #   print(f"{i + 1}: {word}")


if __name__ == '__main__':
    main()
