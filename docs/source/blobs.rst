Blob features
=============

This is a three step process:

1. :ref:`find blobs<Find blobs>`
2. :ref:`sort/filter blobs<Represent blobs>`

Find blobs
----------

Find connected regions (blobs) in a grey-scale image.

.. autoclass:: machinevisiontoolbox.ImageBlobs.ImageBlobsMixin
   :members:

Represent blobs
---------------

This returns a blob feature instance.

.. autoclass:: machinevisiontoolbox.ImageBlobs.Blob
   :members:
   :special-members: __len__, __getitem__