
import os

import numpy as np


def rgb_to_od(image):
    assert image.dtype == np.uint8

    OD = -np.log((image.astype(np.float32) + 1) / 255.0)
    return OD

def normalize_matrix_cols(A):
    return A / np.linalg.norm(A, axis=0)[None, :]

class VahadaneStainDeconvolution:
    def __init__(
        self,
        optical_density_threshold=0.15,
        sparsity_regularizer=1.0,
        regularizer_lasso=0.01,
        background_intensity=245, # FIXME: fit background
        stain_matrix_target_od=np.array(
            [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]]
        ),
        max_c_target=np.array([1.9705, 1.0308])
    ) -> None:
        self.optical_density_threshold = optical_density_threshold
        self.sparsity_regularizer = sparsity_regularizer
        self.regularizer_lasso = regularizer_lasso
        self.background_intensity = background_intensity
        self.stain_matrix_target_od = stain_matrix_target_od
        self.max_c_target = max_c_target

    def fit(self, image):
        stain_matrix = self._estimate_stain_vectors(image)

        C = self._estimate_pixel_concentrations(
            image=image, stain_matrix=stain_matrix
        )

        max_C = np.percentile(C, 99, axis=0).reshape((1, 2))

        self.stain_matrix_target_od = stain_matrix
        self.max_c_target = max_C

    def _estimate_stain_vectors(self, image):
        import spams

        image_od = rgb_to_od(image)
        image_od = image_od.reshape((-1, 3))
        OD = image_od[np.all(image_od > self.optical_density_threshold, axis=1)]

        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        dictionary = spams.trainDL(
            X=OD.T,
            K=2,
            lambda1=self.sparsity_regularizer,
            mode=2,
            modeD=0,
            posAlpha=True,
            posD=True,
            verbose=False
        )
        dictionary = normalize_matrix_cols(dictionary)
        if dictionary[0, 0] > dictionary[1, 0]:
            dictionary = dictionary[:, [1, 0]]
        return dictionary

    def _estimate_pixel_concentrations(self, image, stain_matrix):
        import spams

        image_OD = rgb_to_od(image).reshape((-1, 3))
        lamb = self.regularizer_lasso
        C = (
            spams.lasso(X=image_OD.T, D=stain_matrix, mode=2, lambda1=lamb, pos=True)
            .toarray()
            .T
        )
        return C

    def _reconstruct_image(self, pixel_intensities, kind):
        max_c = np.percentile(pixel_intensities, 99, axis=0).reshape((1, 2))
        pixel_intensities *= self.max_c_target / max_c

        im = np.exp(
            -self.stain_matrix_target_od[:, kind].reshape((-1, 1))
            @ pixel_intensities[:, kind].reshape((-1, 1)).T
        )

        im *= self.background_intensity
        im = np.clip(im, a_min=0, a_max=255)
        im = im.T.astype(np.uint8)
        return im

    def F(self, image, method):
        stain_matrix = self._estimate_stain_vectors(image=image)
        C = self._estimate_pixel_concentrations(image, stain_matrix)

        im_reconstructed = self._reconstruct_image(
            pixel_intensities=C,
            kind={'hematoxylin': 0, 'eosin': 1}[method.lower()]
        )
        return im_reconstructed

def inverted_gray_scale(image):
    from skimage.color import rgb2gray
    from skimage import img_as_ubyte

    assert image.dtype == np.uint8

    image_gray = rgb2gray(image)
    image_gray_inverted = 1 - image_gray
    image_gray_inverted = img_as_ubyte(image_gray_inverted)
    return image_gray_inverted
