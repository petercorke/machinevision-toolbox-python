#!/usr/bin/env python3

from collections import Counter
from machinevisiontoolbox.ImagePointFeatures import BaseFeature2D
import numpy as np
import cv2 as cv


class BagOfWords:

    def __init__(self, images, k=2_000, nstopwords=50, attempts=1):
        """
        Bag of words class

        :param images: a sequence of images or image features
        :type images: Image sequence, iterator returning Image, BaseFeature2D
        :param k: number of visual words, defaults to 2_000
        :type k: int, optional
        :param nstopwords: number of stop words, defaults to 50
        :type nstopwords: int, optional
        :param attempts: number of k-means attempts, defaults to 1
        :type attempts: int, optional

        Creates a bag of words from a set of image features.

        - ``BagOfWords(images)`` will extract features from the image set using
          the SIFT detector.
        - ``BagOfWords(features)`` will use the passed image ``features`` which
          must have a valid ``.id`` attribute indicating which image in the bag
          they belong to.

        k-means clustering is performed to assign a word label to every feature.

        ``.words`` is an array of word labels that corresponds to the array of
        image features ``.features``.  The word labels are integers, initially
        in the range 0 to ``.k``.

        The cluster centroids are retained as a k x N array ``.centroids`` with
        one row per word centroid and each row is a feature descriptor, 128
        elements long in the case of SIFT.

        Stop words are those visual words that occur most often and we can
        remove ``nstopwords`` of them. In this case number of visual words is
        reduced and the word labels are in the range 0 to ``.nstop`` where
        ``.nstop`` is less than ``.k``.

        The centroids are reordered so that the last ``.nstopwords`` rows
        correspond to the stop words.  When a new set of image features is
        assigned labels from the ``.centroids`` any with a label greater that
        ``.nstop`` is a stop word and can be discarded.

        :reference: J.Sivic and A.Zisserman, "Video Google: a text retrieval
            approach to object matching in videos", in Proc. Ninth IEEE Int.
            Conf. on Computer Vision, pp.1470-1477, Oct. 2003.

        :seealso: `cv2.kmeans <https://docs.opencv.org/master/d5/d38/group__core__cluster.html#ga9a34dc06c6ec9460e90860f15bcd2f88>`_
            :meth:`.SIFT`
            :meth:`.recall`
        """

        if images is None:
            return

        if isinstance(images, BaseFeature2D):
            # passed the features
            features = images
        else:
            # passed images, compute the features
            features = None
            for image in images:
                if features is None:
                    features = image.SIFT()
                else:
                    features += image.SIFT()
            self._images = images

        # save the image id's 
        self._image_id = np.r_[features.id]
        self._nimages = self._image_id.max() + 1
        self._features = features

        # do the clustering
        # NO IDEA WHAT EPSILON ACTUALLY MEANS, NORM OF THE SHIFT IN CENTROIDS?
        # NO IDEA HOW TO TELL WHAT CRITERIA IT TERMINATES ON
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        ret, labels, centroids = cv.kmeans(
                        data=features._descriptor, 
                        K=k,
                        bestLabels=None,
                        criteria=criteria,
                        attempts=attempts,
                        flags=cv.KMEANS_RANDOM_CENTERS)

        self._k = k

        self._words = labels.ravel()
        self._centroids = centroids
        self._word_freq_vectors = None

        self._nstopwords = nstopwords
        if nstopwords > 0:
            self._remove_stopwords()

    @property
    def nimages(self):
        """
        Number of images associated with this bag

        :return: number of images
        :rtype: int
        """
        return self._images

    @property
    def images(self):
        """
        Images associated with this bag

        :return: images associated with this bag
        :rtype: Image sequence or iterator

        Only valid if the bag was constructed from images rather than features.
        """
        return self._images

    @property
    def k(self):
        """
        Number of words in the visual vocabulary

        :return: number of words
        :rtype: int
        """
        return self._k

    @property
    def words(self):
        """
        Word labels for every feature

        :return: word labels
        :rtype: ndarray(N)

        Word labels are arranged such that the top ``nstopwords`` labels
        """
        return self._words

    @property
    def nwords(self):
        """
        Number of usable words

        :return: number of usable words
        :rtype: int
        """
        return self._k - self._nstopwords

    @property
    def nstopwords(self):
        """
        Number of stop words

        :return: Number of stop words
        :rtype: int
        """
        return self._nstopwords

    @property
    def nstop(self):

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
        with a label ``>= nstop`` is a stop word.

        We keep the stop words in the centroid array for the recall process.
        """
        return self._centroids


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

        # # top nstop most frequent are the stop words
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
        # now all stop words have an index in the range [k-nstop, k)
        mapdict = {}
        for w in unique_words:
            mapdict[map[w]] = w

        # map the word labels
        words = np.array([mapdict[w] for w in self.words])

        # only retain the non stop words
        keep = words < self.nstop
        self._words = words[keep]
        self._image_id = self._image_id[keep]
        self._features = self._features[keep]

        # rearrange the cluster centroids
        self._centroids = self._centroids[map]

    def recall(self, images):
        # compute the features
        features = None
        for image in images:
            if features is None:
                features = image.SIFT()
            else:
                features += image.SIFT()

        # assign features to given cluster centroids
        bfm = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
        matches = bfm.match(features._descriptor, self._centroids)

        # the elements of matches are:
        #  queryIdx: new feature index
        #  trainingIdx: cluster centre index
        words = np.array([m.trainIdx for m in matches])

        id = np.r_[features.id]

        # only retain the non stop words
        keep = words < self.nstop
        words = words[keep]
        image_id = id[keep]

        bag = BagOfWords()
        bag._k = self.k
        bag._nstopwords = self._nstopwords
        bag._images = images
        bag._nimages = len(images)
        bag._words = words
        bag._image_id = image_id
        bag._centroids = None
        bag._word_freq_vectors = None

        return bag


    def features(self, word):
        #BagOfWords.isword Features from words
        #
        # F = B.isword(W) is a vector of feature objects that are assigned to any of
        # the word W.  If W is a vector of words the result is a vector of features
        # assigned to all the words in W.
        return self._features[self.words == word]

    def occurrence(self, word):
        """
        Number of occurrences of specified word

        :param word: word label
        :type word: int
        :return: total number of times that ``word`` appears in this bag
        :rtype: int
        """
        return np.sum(self.words == word)

    def word_freq_vector(self, i=None):
        """
        Word frequency vector for image

        :param i: image within bag, defaults to None
        :type i: int, optional
        :return: word frequency vector
        :rtype: ndarray(K)

        This is the word-frequency vector for the ``i``'th image in the bag. The
        angle between any two WFVs is an indication of image similarity.

        If ``i`` is None then the word-frequency matrix is returned, where the
        columns are the word-frequency vectors for the images in the bag.

        .. note:: The word vector is expensive to compute so a lazy evaluation
            is performed on the first call to this method.
        """
        self._compute_word_freq_vectors()

        if i is not None:
            return self._word_freq_vectors[:, i]
        else:
            return self._word_freq_vectors

    def iwf(self):
        """
        Image word frequency

        :return: image word frequency matrix
        :rtype: ndarray(M,N)

        Each column corresponds to an image in the bag and rows are the number
        of occurences of that word in that image.
        """
        N = self.nimages  # number of images
        id = np.array(self._image_id)

        nl = self.k - self.nstopwords
        W = []

        for i in range(self.nimages):
            # get the words associated with image i
            words = self.words[id == i]

            # create columns of the W
            unique, unique_counts = np.unique(words, return_counts=True)
            # [w,f] = count_unique(words)
            v = np.zeros((nl,))
            v[unique] = unique_counts
            W.append(v)
        return np.column_stack(W)

    def _compute_word_freq_vectors(self, bag2=None):

        if self._word_freq_vectors is None:

            W = self.iwf()
            N = self.nimages

            # total number of occurences of word i
            Ni = (W > 0).sum(axis=1)

            M = []
            for i in range(self.nimages):
                # number of words in this image
                nd = W[:, i].sum()

                # word occurrence frequency
                nid = W[:, i]

                with np.errstate(divide='ignore', invalid='ignore'):
                    v = nid / nd * np.log(N / Ni)

                v[~np.isfinite(v)] = 0
                
                M.append(v)

                self._word_freq_vectors =  np.column_stack(M)

    def wordfreq(self):

        #BagOfWords.wordfreq Word frequency statistics
        #
        # [W,N] = B.wordfreq[] is a vector of word labels W and the corresponding
        # elements of N are the number of occurrences of that word.
        return np.unique(self.words, return_counts=True)


    def similarity(self, other):
        """
        Compute similarity between two bags of words

        :param other: bag of words
        :type other: BagOfWords
        :return: confusion matrix
        :rtype: ndarray(M,N)

        The array has rows corresponding to the images in ``self`` and 
        columns corresponding to the images in ``other``.

        :seealso: :meth:`.closest`
        """

        sim = np.empty((self.nimages, other.nimages))
        for i in range(self.nimages):
            for j in range(other.nimages):
                v1 = self.word_freq_vector(i)
                v2 = other.word_freq_vector(j)
                with np.errstate(divide='ignore', invalid='ignore'):
                    sim[i, j] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return sim

    def closest(self, S, i):
        """
        Find closest image

        :param S: bag similarity matrix
        :type S: ndarray(N,M)
        :param i: the query image index
        :type i: int
        :return: index of the recalled image and similarity
        :rtype: int, float

        :seealso: :meth:`.similarity`
        """
        s = S[:, i]
        index = np.argsort(-s)
        
        return index, s[index]

    def contains(self, word):
        """
        Images that contain specified word

        :param word: word label
        :type word: int
        :return: list of images containing this word
        :rtype: list
        """
        return self._image_id[self.words == word]
            
    def exemplars(self, word, images=None, maxperimage=2, columns=10, max=None, width=50, **kwargs):
        """
        Composite image containing exemplars of specified word

        :param word: word label
        :type word: int
        :param images: the set of images corresponding to this bag, only 
            required if the bag was constructed from features not images.
        :param maxperimage: maximum number of exemplars drawn from any one image, defaults to 2
        :type maxperimage: int, optional
        :param columns: number of exemplar images in each row, defaults to 10
        :type columns: int, optional
        :param max: maximum number of exemplar images, defaults to None
        :type max: int, optional
        :param width: , defaults to 50
        :type width: int, optional
        :return: composite image
        :rtype: Image

        Produces a grid of examples of a particular visual word.

        :seealso: :meth:`BaseFeature2D.support` :meth:`Image.Tile`
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
            if len(exemplars) >= max:
                break

        return Image.Tile(exemplars, columns=columns, **kwargs)

