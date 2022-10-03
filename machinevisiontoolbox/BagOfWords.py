#!/usr/bin/env python3

from collections import Counter
from machinevisiontoolbox.ImagePointFeatures import BaseFeature2D
import numpy as np
import cv2 as cv

# TODO: remove top N% and bottom M% of words by frequency
class BagOfWords:

    def __init__(self, images, k=2_000, nstopwords=0, attempts=1, seed=None):
        r"""
        Bag of words class

        :param images: a sequence of images or set of image features
        :type images: :class:`~machinevisiontoolbox.Image` iterable, :class:`~machinevisiontoolbox.PointFeatures.BaseFeature2D`
        :param k: number of visual words, defaults to 2000
        :type k: int, optional
        :param nstopwords: number of stop words, defaults to 50
        :type nstopwords: int, optional
        :param attempts: number of k-means attempts, defaults to 1
        :type attempts: int, optional

        Bag of words is a powerful feature-based method for matching images
        from widely different viewpoints. 
        
        This class creates a bag of words from a sequence of images or a set of
        point features.  In the former case, the features will have an ``.id``
        equal to the index of the image in the sequence.  For the latter case,
        features must have a valid ``.id`` attribute indicating which image in
        the bag they belong to.

        k-means clustering is performed to assign a word label to every feature.
        The cluster centroids are retained as a :math:`k \times N` array
        ``.centroids`` with one row per word centroid and each row is a feature
        descriptor, 128 elements long in the case of SIFT.

        ``.words`` is an array of word labels that corresponds to the array of
        image features ``.features``.  The word labels are integers, initially
        in the range [0, ``k``).

        Stop words are those visual words that occur most often and we can
        remove ``nstopwords`` of them. The centroids are reordered so that the
        last ``nstopwords`` rows correspond to the stop words.  When a new set
        of image features is assigned labels from the ``.centroids`` any with a
        label greater that ``.nstopwords`` is a stop word and can be discarded.

        :reference: 
            - Video Google: a text retrieval approach to object matching in videos
              J.Sivic and A.Zisserman, 
              in Proc. Ninth IEEE Int. Conf. on Computer Vision, 
              pp.1470-1477, Oct. 2003.
            - Robotics, Vision & Control for Python, Section 12.4.2, 
                P. Corke, Springer 2023.

        :seealso: :meth:`recall` :meth:`~machinevisiontoolbox.ImagePointFeatures.BaseFeature2D`
            :meth:`~machinevisiontoolbox.ImagePointFeatures.SIFT`
            `cv2.kmeans <https://docs.opencv.org/master/d5/d38/group__core__cluster.html#ga9a34dc06c6ec9460e90860f15bcd2f88>`_
        """

        if images is None:
            return

        if isinstance(images, BaseFeature2D):
            # passed the features
            features = images
        else:
            # passed images, compute the features
            features = []
            for image in images:
                features += image.SIFT()
        features.sort(by="scale", inplace=True)

        self._images = images

        # save the image id's 
        self._image_id = np.r_[features.id]
        self._nimages = self._image_id.max() + 1
        self._features = features

        # do the clustering
        # NO IDEA WHAT EPSILON ACTUALLY MEANS, NORM OF THE SHIFT IN CENTROIDS?
        # NO IDEA HOW TO TELL WHAT CRITERIA IT TERMINATES ON
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        if seed is not None:
            cv.setRNGSeed(seed)

        ret, labels, centroids = cv.kmeans(
                        data=features._descriptor, 
                        K=k,
                        bestLabels=None,
                        criteria=criteria,
                        attempts=attempts,
                        flags=cv.KMEANS_RANDOM_CENTERS)

        self._k = k

        self._words = labels.ravel()
        self._labels = labels.ravel()
        self._centroids = centroids
        self._word_freq_vectors = None

        self._nstopwords = nstopwords
        if nstopwords > 0:
            self._remove_stopwords()

        # compute word frequency vectors

        maxwords = self.k - self.nstopwords

        W = []
        id = np.array(self._features.id)
        for i in range(self.nimages):
            # get the words associated with image i
            words = self.words[id == i]

            # create columns of the W
            v = BagOfWords._word_freq_vector(words, maxwords)
            W.append(v)
        W = np.column_stack(W)

        N = self.nimages

        # total number of occurences of word i
        # multiple occurences in the one image count only as one
        ni = (W > 0).sum(axis=1)
        idf = np.log(N / ni)

        M = []
        
        for i in range(self.nimages):
            # number of words in this image
            nd = W[:, i].sum()

            # word occurrence frequency
            nid = W[:, i]

            with np.errstate(divide='ignore', invalid='ignore'):
                v = nid / nd * idf

            v[~np.isfinite(v)] = 0
            M.append(v)

        self._word_freq_vectors =  np.column_stack(M)
        self._idf = idf

    def wwfv(self, i=None):
        """
        Weighted word frequency vector for image

        :param i: image within bag, defaults to all images
        :type i: int, optional
        :return: word frequency vector or vectors
        :rtype: ndarray(K), ndarray(N,K)

        This is the word-frequency vector for the ``i``'th image in the bag. The
        angle between any two WFVs is an indication of image similarity.

        If ``i`` is None then the word-frequency matrix is returned, where the
        columns are the word-frequency vectors for the images in the bag.

        .. note:: The word vector is expensive to compute so a lazy evaluation
            is performed on the first call to this method.
        """
        if i is not None:
            v = self._word_freq_vectors[:, i]
            if v.ndim == 1:
                return np.c_[v]
        else:
            return self._word_freq_vectors

    @property
    def nimages(self):
        """
        Number of images associated in the bag

        :return: number of images
        :rtype: int
        """
        return self._nimages

    @property
    def images(self):
        """
        Images associated with this bag

        :return: images associated with this bag
        :rtype: :class:`~machinevisiontoolbox.Image` iterable

        .. note:: Only valid if the bag was constructed from images rather than features.
        """
        return self._images

    @property
    def k(self):
        """
        Number of words in the visual vocabulary

        :return: number of words
        :rtype: int

        :seealso: :meth:`nstopwords`
        """
        return self._k

    @property
    def words(self):
        """
        Word labels for every feature

        :return: word labels
        :rtype: ndarray(N)

        Word labels are arranged such that the top ``nstopwords`` labels are
        stop words.

        :seealso: :meth:`nstopwords`
        """
        return self._words

    # TODO better name for above

    def word(self, f):
        """
        Word labels for original feature

        :return: word labels
        :rtype: ndarray(N)

        Word labels are arranged such that the top ``nstopwords`` labels
        """
        return self._labels[f]

    @property
    def nwords(self):
        """
        Number of usable words

        :return: number of usable words
        :rtype: int

        This is ``k`` - ``nstopwords``.

        :seealso: :meth:`k` :meth:`nstopwords`
        """
        return self._k - self._nstopwords

    @property
    def nstopwords(self):
        """
        Number of stop words

        :return: Number of stop words
        :rtype: int

        :seealso: :meth:`k` :meth:`nwords`
        """
        return self._nstopwords

    @property
    def firststop(self):
        """
        First stop word

        :return: word index of first stop word
        :rtype: int
        """
        return self.k - self._nstopwords

    @property
    def centroids(self):
        """
        Word feature centroids

        :return: centroids of visual word features
        :rtype: ndarray(k,N)

        Is an array with one row per visual word, and the row is the feature
        descriptor vector.  eg. for SIFT features it is 128 elements.

        Centroids are arranged such that the last ``nstopwords`` rows correspond
        to the stop words.  After clustering against the centroids, any word
        with a label ``>= nstopwords`` is a stop word.

        .. note:: The stop words are kept in the centroid array for the recall process.

        :seealso: :meth:`similarity`
        """
        return self._centroids

    def __repr__(self):
        return str(self)

    def __str__(self):
        s = f"BagOfWords: {len(self.words)} features from {self.nimages} images"
        s += f", {self.nwords} words, {self.nstopwords} stop words"
        return s

    def _remove_stopwords(self, verbose=True):
        #BagOfWords.remove_stop Remove stop words
        #
        # B.remove_stop(N) removes the N most frequent words (the stop words)
        # from the self.  All remaining words are renumbered so that the word
        # labels are consecutive.

        # words, freq = self.wordfreq()
        # index = np.argsort(-freq)  # sort descending order

        # # top ``nstopwords`` most frequent are the stop words
        # stopwords = words[index[:self._nstopwords]]

        unique_words, freq = self.wordfreq()

        # unique_words will be [0,k)
        index = np.argsort(-freq)  # sort descending order
        stopwords = unique_words[index[:self.nstopwords]]  # array of stop words

        stopfeatures = freq[stopwords].sum()
        print(f"Removing {stopfeatures} features ({stopfeatures/len(self.words) * 100:.1f}%) associated with {self.nstopwords} most frequent words")

        k = np.full(index.shape, False, dtype=bool)
        k[stopwords] = True
        # k = freq > stop_cut  # index of all the stop words
        # indices of all non-stop words, followed by all stop words
        map = np.hstack((unique_words[~k], unique_words[k]))
        # create a dictionary from old label to new label
        # now all stop words have an index in the range [k-nstopwords, k)
        mapdict = {}
        for w in unique_words:
            mapdict[map[w]] = w

        # map the word labels
        words = np.array([mapdict[w] for w in self.words])

        self._labels = words
        
        # only retain the non stop words
        keep = words < self.nstopwords
        self._words = words[keep]
        self._image_id = self._image_id[keep]
        self._features = self._features[keep]

        # rearrange the cluster centroids
        self._centroids = self._centroids[map]

    def similarity(self, arg):
        """
        Compute similarity between bag and query images

        :param other: bag of words
        :type other: BagOfWords
        :return: confusion matrix
        :rtype: ndarray(M,N)

        The array has rows corresponding to the images in ``self`` and 
        columns corresponding to the images in ``other``.

        :seealso: :meth:`.closest`
        """
        if isinstance(arg, np.ndarray):
            wwfv = arg
            sim = np.empty((wwfv.shape[1], self.nimages))
            
            for j, vj in enumerate(wwfv.T):
                for i in range(self.nimages):
                    vi = self.wwfv(i)

                    with np.errstate(divide='ignore', invalid='ignore'):
                        sim[j, i] = np.dot(vi.ravel(), vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
        else:
            images = arg
            if not hasattr(images, '__iter__'):
                # if not iterable like a FileCollection or VideoFile turn the image
                # into a list of 1
                images = [images]

            # similarity has bag index as column, query index as row
            sim = np.empty((len(images), self.nimages))
            for j, image in enumerate(images):
                features = image.SIFT(id='image')

                # assign features to given cluster centroids
                # the elements of matches are:
                #  queryIdx: new feature index
                #  trainingIdx: cluster centre index
                bfm = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
                matches = bfm.match(features._descriptor, self._centroids)
                words = np.array([m.trainIdx for m in matches])

                keep = words < self.nstopwords
                words = words[keep]

                # word occurrence frequency
                nid = BagOfWords._word_freq_vector(words, self.k - self.nstopwords)
            
                # number of words in this image
                nd = nid.sum()

                with np.errstate(divide='ignore', invalid='ignore'):
                    v2 = nid / nd * self._idf

                v2[~np.isfinite(v2)] = 0

                for i in range(self.nimages):
                    v1 = self.wwfv(i).ravel()

                    with np.errstate(divide='ignore', invalid='ignore'):
                        sim[j, i] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        if sim.shape[0] == 1:
            sim = sim[0, :]
        return sim

    def retrieve(self, images):

        S = self.similarity(images).ravel()
        k = np.argmax(S)
        return k, S[k]

    def features(self, word):
        """
        Get features corresponding to word

        :param word: visual word label
        :type word: int
        :return: features corresponding to this label
        :rtype: :class:`~machinevisiontoolbox.PointFeatures.BaseFeature2D`

        Return a slice of the image features corresponding to this word label.
        The ``.id`` attribute of each feature indicates which image in the bag
        it belongs to.
        """
        return self._features[self.words == word]

    def occurrence(self, word):
        """
        Number of occurrences of specified word

        :param word: visual word label
        :type word: int
        :return: total number of times that visual ``word`` appears in this bag
        :rtype: int
        """
        return np.sum(self.words == word)

    @staticmethod
    def _word_freq_vector(words, maxwords):
        # create columns of the W
        unique, unique_counts = np.unique(words, return_counts=True)
        # [w,f] = count_unique(words)
        v = np.zeros((maxwords,))
        v[unique] = unique_counts
        return v

    def wordfreq(self):
        """
        Get visual word frequency

        :return: visual words, visual word frequency
        :rtype: ndarray, ndarray

        Returns two arrays, one containing all visual words, the other containing
        the frequency of the corresponding word across all images.
        """

        #BagOfWords.wordfreq Word frequency statistics
        #
        # [W,N] = B.wordfreq[] is a vector of word labels W and the corresponding
        # elements of N are the number of occurrences of that word.
        return np.unique(self.words, return_counts=True)

    def closest(self, S, i):
        """
        Find closest image

        :param S: bag similarity matrix
        :type S: ndarray(N,M)
        :param i: the query image index
        :type i: int
        :return: index of the recalled image and similarity
        :rtype: int, float

        :seealso: :meth:`similarity`
        """
        s = S[:, i]
        index = np.argsort(-s)
        
        return index, s[index]

    def contains(self, word):
        """
        Images that contain specified word

        :param word: visual word label
        :type word: int
        :return: list of images containing this word
        :rtype: list

        :seealso: :meth:`exemplars`
        """
        return np.unique(self._image_id[self.words == word])
            

    def exemplars(self, word, images=None, maxperimage=2, columns=10, max=None, width=50, **kwargs):
        """
        Composite image containing exemplars of specified word

        :param word: visual word label
        :type word: int
        :param images: the set of images corresponding to this bag, only 
            required if the bag was constructed from features not images.
        :param maxperimage: maximum number of exemplars drawn from any one image, defaults to 2
        :type maxperimage: int, optional
        :param columns: number of exemplar images in each row, defaults to 10
        :type columns: int, optional
        :param max: maximum number of exemplar images, defaults to None
        :type max: int, optional
        :param width: width of image thumbnail, defaults to 50
        :type width: int, optional
        :return: composite image
        :rtype: :class:`~machinevisiontoolbox.Image`

        Produces a grid of examples of a particular visual word.

        :seealso: :meth:`contains` 
            :meth:`~machinevisiontoolbox.ImagePointFeatures.BaseFeature2D.support` 
            :meth:`~machinevisiontoolbox.Image.Tile`
        """
        from machinevisiontoolbox import Image

        exemplars = []
        count = Counter()
        if images is None:
            images = self._images
        for feature in self.features(word):
            count[feature.id] += 1

            if count[feature.id] > maxperimage:
                continue

            exemplars.append(feature.support(images, width))
            if max is not None and len(exemplars) >= max:
                break

        return Image.Tile(exemplars, columns=columns, **kwargs)

if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    from machinevisiontoolbox import *
    import cv2 as cv

    cv.setRNGSeed(0)

    images = ImageCollection('campus/*.png', mono=True)

    features = []
    for image in images:
        features += image.SIFT()
    # sort them in descending order by strength
    features.sort(by="scale", inplace=True)

    features[:10].table()

    ex = []
    for i in range(400):
        ex.append(features[i].support(images))

    Image.Tile(ex, columns=20).disp(plain=True)


    feature = features[108]
    print(feature)

    bag = BagOfWords(features, 2_000)

    w = bag.word(108)
    print(w)
    print(bag.occurrence(w))
    print(bag.contains(w))

    bag.exemplars(w, images)

    bag = BagOfWords(images, 2_000)
    print(bag)

    w, f = bag.wordfreq()
    print(len(w))

    bag = BagOfWords(images, 2_000, nstopwords=50)
    print(bag)

    print(bag.wwfv(0).shape)
    print(bag.wwfv().shape)

    print(bag.similarity(bag.wwfv(3)))
    print(bag.similarity(images[:5]))

    sim_8 = bag.similarity(images[8]).ravel()
    print(sim_8)
    k = np.argsort(-sim_8);
    print(np.c_[sim_8[k], k])

    ss = []
    for i in range(4):
        ss.append(images[k[i]])
    Image.Tile(ss, columns=2).disp()

    holdout = ImageCollection("campus/holdout/*.png", mono=True);

    sim = bag.similarity(holdout)

    sim_2 = bag.similarity(holdout[2]).ravel()
    print(sim_2)
    k = np.argsort(-sim_2);
    print(np.c_[sim_2[k], k])

    ss = [holdout[2]]
    for i in range(3):
        ss.append(images[k[i]])
    Image.Tile(ss, columns=2).disp()

    Image(sim).disp(block=True)



    