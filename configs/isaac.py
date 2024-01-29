__copyright__ = "Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import io
import json
from typing import List
import numpy as np
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry, orchestrator
from .tools import colorize_normals
import cv2 
import os

__version__ = "0.0.1"


class BasicWriter(Writer):
    """Basic writer capable of writing built-in annotator groundtruth.

    Attributes:
        output_dir:
            Output directory string that indicates the directory to save the results.
        semantic_types:
            List of semantic types to consider when filtering annotator data. Default: ["class"]
        rgb:
            Boolean value that indicates whether the rgb annotator will be activated
            and the data will be written or not. Default: False.
        bounding_box_2d_tight:
            Boolean value that indicates whether the bounding_box_2d_tight annotator will be activated
            and the data will be written or not. Default: False.
        bounding_box_2d_loose:
            Boolean value that indicates whether the bounding_box_2d_loose annotator will be activated
            and the data will be written or not. Default: False.
        semantic_segmentation:
            Boolean value that indicates whether the semantic_segmentation annotator will be activated
            and the data will be written or not. Default: False.
        instance_id_segmentation:
            Boolean value that indicates whether the instance_id_segmentation annotator will be activated
            and the data will be written or not. Default: False.
        instance_segmentation:
            Boolean value that indicates whether the instance_segmentation annotator will be activated
            and the data will be written or not. Default: False.
        distance_to_camera:
            Boolean value that indicates whether the distance_to_camera annotator will be activated
            and the data will be written or not. Default: False.
        distance_to_image_plane:
            Boolean value that indicates whether the distance_to_image_plane annotator will be activated
            and the data will be written or not. Default: False.
        bounding_box_3d:
            Boolean value that indicates whether the bounding_box_3d annotator will be activated
            and the data will be written or not. Default: False.
        occlusion:
            Boolean value that indicates whether the occlusion annotator will be activated
            and the data will be written or not. Default: False.
        normals:
            Boolean value that indicates whether the normals annotator will be activated
            and the data will be written or not. Default: False.
        motion_vectors:
            Boolean value that indicates whether the motion_vectors annotator will be activated
            and the data will be written or not. Default: False.
        camera_params:
            Boolean value that indicates whether the camera_params annotator will be activated
            and the data will be written or not. Default: False.
        pointcloud:
            Boolean value that indicates whether the pointcloud annotator will be activated
            and the data will be written or not. Default: False.
        image_output_format:
            String that indicates the format of saved RGB images. Default: "png"
        colorize_semantic_segmentation:
            If ``True``, semantic segmentation is converted to an image where semantic IDs are mapped to colors
            and saved as a uint8 4 channel PNG image. If ``False``, the output is saved as a uint32 PNG image.
            Defaults to ``True``.
        colorize_instance_id_segmentation:
            If True, instance id segmentation is converted to an image where instance IDs are mapped to colors.
            and saved as a uint8 4 channel PNG image. If ``False``, the output is saved as a uint32 PNG image.
            Defaults to ``True``.
        colorize_instance_segmentation:
            If True, instance segmentation is converted to an image where instance are mapped to colors.
            and saved as a uint8 4 channel PNG image. If ``False``, the output is saved as a uint32 PNG image.
            Defaults to ``True``.
        frame_padding:
            Pad the frame number with leading zeroes.  Default: 4

    Example:
        >>> import omni.replicator.core as rep
        >>> camera = rep.create.camera()
        >>> render_product = rep.create.render_product(camera, (1024, 1024))
        >>> writer = rep.WriterRegistry.get("BasicWriter")
        >>> import carb
        >>> tmp_dir = carb.tokens.get_tokens_interface().resolve("${temp}/rgb")
        >>> writer.initialize(output_dir=tmp_dir, rgb=True)
        >>> writer.attach([render_product])
        >>> rep.orchestrator.run()
    """

    def __init__(
        self,
        output_dir: str,
        semantic_types: List[str] = None,
        rgb: bool = False,
        bounding_box_2d_tight: bool = False,
        bounding_box_2d_loose: bool = False,
        semantic_segmentation: bool = False,
        instance_id_segmentation: bool = False,
        instance_segmentation: bool = False,
        # background_rand: bool = False,
        distance_to_camera: bool = False,
        distance_to_image_plane: bool = False,
        bounding_box_3d: bool = False,
        occlusion: bool = False,
        normals: bool = False,
        motion_vectors: bool = False,
        camera_params: bool = False,
        pointcloud: bool = False,
        image_output_format: str = "jpg",
        colorize_semantic_segmentation: bool = True,
        colorize_instance_id_segmentation: bool = True,
        colorize_instance_segmentation: bool = True,
        skeleton_data: bool = False,
        frame_padding: int = 4,
        depth_max:float = 1_500,
        scale_val:int = 150_000,
        missing_val:int = np.iinfo(np.int32).max,
    ):
        self._output_dir = output_dir
        self.backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self._backend = self.backend  # Kept for backwards compatibility
        self._output_dir = self.backend.output_dir
        self._frame_id = 0
        self._sequence_id = 0
        self._image_output_format = image_output_format
        self._output_data_format = {}
        self.annotators = []
        self.version = __version__
        self._frame_padding = frame_padding

        self.colorize_semantic_segmentation = colorize_semantic_segmentation
        self.colorize_instance_id_segmentation = colorize_instance_id_segmentation
        self.colorize_instance_segmentation = colorize_instance_segmentation

        self.depth_max = depth_max
        self.scale_val =  scale_val
        self.missing_val = missing_val

        # Specify the semantic types that will be included in output
        if semantic_types is None:
            semantic_types = ["class"]

        # RGB
        if rgb:
            self.annotators.append(
                AnnotatorRegistry.get_annotator("rgb")
            )

        # Bounding Box 2D
        if bounding_box_2d_tight:
            self.annotators.append(
                AnnotatorRegistry.get_annotator("bounding_box_2d_tight", init_params={"semanticTypes": semantic_types})
            )

        if bounding_box_2d_loose:
            self.annotators.append(
                AnnotatorRegistry.get_annotator("bounding_box_2d_loose", init_params={"semanticTypes": semantic_types})
            )

        # Semantic Segmentation
        if semantic_segmentation:
            self.annotators.append(
                AnnotatorRegistry.get_annotator(
                    "semantic_segmentation",
                    init_params={"semanticTypes": semantic_types, "colorize": colorize_semantic_segmentation}
                )
            )

        # Instance Segmentation
        if instance_id_segmentation:
            self.annotators.append(
                AnnotatorRegistry.get_annotator(
                    "instance_id_segmentation",
                    init_params={"colorize": colorize_instance_id_segmentation}
                )
            )

        # Instance Segmentation
        if instance_segmentation:
            self.annotators.append(
                AnnotatorRegistry.get_annotator(
                    "instance_segmentation",
                    init_params={"semanticTypes": semantic_types, "colorize": colorize_instance_segmentation}
                )
            )


        # # Background Rand
        # if background_rand:
        #     self.annotators.append(AnnotatorRegistry.get_annotator("background_rand", init_params={"colorize": True}))

        # Depth
        if distance_to_camera:
            self.annotators.append(
                AnnotatorRegistry.get_annotator("distance_to_camera")
            )

        if distance_to_image_plane:
            self.annotators.append(
                AnnotatorRegistry.get_annotator("distance_to_image_plane")
            )

        # Bounding Box 3D
        if bounding_box_3d:
            self.annotators.append(
                AnnotatorRegistry.get_annotator(
                    "bounding_box_3d",
                    init_params={"semanticTypes": semantic_types}
                )
            )

        # Motion Vectors
        if motion_vectors:
            self.annotators.append(
                AnnotatorRegistry.get_annotator("motion_vectors")
            )

        # Occlusion
        if occlusion:
            self.annotators.append(
                AnnotatorRegistry.get_annotator("occlusion")
            )

        # Normals
        if normals:
            self.annotators.append(
                AnnotatorRegistry.get_annotator("normals")
            )

        # Camera Params
        if camera_params:
            self.annotators.append(
                AnnotatorRegistry.get_annotator("camera_params")
            )

        # Pointcloud
        if pointcloud:
            self.annotators.append(AnnotatorRegistry.get_annotator("pointcloud"))

        # Skeleton Data
        if skeleton_data:
            self.annotators.append(AnnotatorRegistry.get_annotator("skeleton_data"))

    def write(self, data: dict):
        """Write function called from the OgnWriter node on every frame to process annotator output.

        Args:
            data: A dictionary containing the annotator data for the current frame.
        """
        # Check for on_time triggers
        # For each on_time trigger, prefix the output frame number with the trigger counts
        sequence_id = ""
        for trigger_name, call_count in data["trigger_outputs"].items():
            if "on_time" in trigger_name:
                sequence_id = f"{call_count}_{sequence_id}"
        if sequence_id != self._sequence_id:
            self._frame_id = 0
            self._sequence_id = sequence_id

        for annotator in data.keys():
            annotator_split = annotator.split("-")
            render_product_path = ""
            multi_render_prod = 0
            # multiple render_products
            if len(annotator_split) > 1:
                multi_render_prod = 1
                render_product_name = annotator_split[-1]
                render_product_path = f"{render_product_name}/"

            if annotator.startswith("rgb"):
                if multi_render_prod:
                    render_product_path += "rgb/"
                self._write_rgb(data, render_product_path, annotator)

            if annotator.startswith("normals"):
                if multi_render_prod:
                    render_product_path += "normals/"
                self._write_normals(data, render_product_path, annotator)

            if annotator.startswith("distance_to_camera"):
                if multi_render_prod:
                    render_product_path += "distance_to_camera/"
                self._write_distance_to_camera(data, render_product_path, annotator)

            if annotator.startswith("distance_to_image_plane"):
                if multi_render_prod:
                    render_product_path += "distance_to_image_plane/"
                self._write_distance_to_image_plane(data, render_product_path, annotator)

            if annotator.startswith("semantic_segmentation"):
                if multi_render_prod:
                    render_product_path += "semantic_segmentation/"
                self._write_semantic_segmentation(data, render_product_path, annotator)

            if annotator.startswith("instance_id_segmentation"):
                if multi_render_prod:
                    render_product_path += "instance_id_segmentation/"
                self._write_instance_id_segmentation(data, render_product_path, annotator)

            if annotator.startswith("instance_segmentation"):
                if multi_render_prod:
                    render_product_path += "instance_segmentation/"
                self._write_instance_segmentation(data, render_product_path, annotator)

            if annotator.startswith("motion_vectors"):
                if multi_render_prod:
                    render_product_path += "motion_vectors/"
                self._write_motion_vectors(data, render_product_path, annotator)

            if annotator.startswith("occlusion"):
                if multi_render_prod:
                    render_product_path += "occlusion/"
                self._write_occlusion(data, render_product_path, annotator)

            if annotator.startswith("bounding_box_3d"):
                if multi_render_prod:
                    render_product_path += "bounding_box_3d/"
                self._write_bounding_box_data(data, "3d", render_product_path, annotator)

            if annotator.startswith("bounding_box_2d_loose"):
                if multi_render_prod:
                    render_product_path += "bounding_box_2d_loose/"
                self._write_bounding_box_data(data, "2d_loose", render_product_path, annotator)

            if annotator.startswith("bounding_box_2d_tight"):
                if multi_render_prod:
                    render_product_path += "bounding_box_2d_tight/"
                self._write_bounding_box_data(data, "2d_tight", render_product_path, annotator)

            if annotator.startswith("camera_params"):
                if multi_render_prod:
                    render_product_path += "camera_params/"
                self._write_camera_params(data, render_product_path, annotator)

            if annotator.startswith("pointcloud"):
                if multi_render_prod:
                    render_product_path += "pointcloud/"
                self._write_pointcloud(data, render_product_path, annotator)

            if annotator.startswith("skeleton_data"):
                if multi_render_prod:
                    render_product_path += "skeleton_data/"
                self._write_skeleton(data, render_product_path, annotator)

        self._frame_id += 1

    def _write_rgb(self, data: dict, render_product_path: str, annotator: str):
        file_path = f"{render_product_path}rgb_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.{self._image_output_format}"
        from PIL import Image 
        import os
        png = Image.fromarray(data[annotator])
        png.convert("RGB").save(os.path.join(self._output_dir, file_path))
        print(os.path.join(self._output_dir, file_path))
        #self._backend.write_image(file_path, data[annotator])

    # def _write_rgb(self, data: dict, render_product_path: str, annotator: str):
    #     file_path = f"{render_product_path}rgb_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.{self._image_output_format}.png"
    #     self._backend.write_image(file_path, data[annotator])


    def _write_normals(self, data: dict, render_product_path: str, annotator: str):
        normals_data = data[annotator]
        file_path = f"{render_product_path}normals_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
        colorized_normals_data = colorize_normals(normals_data)
        self._backend.write_image(file_path, colorized_normals_data)

    def split_png(self,depth):
        missing_mask = depth > self.depth_max
        depth = (depth / self.depth_max) * self.scale_val
        depth[missing_mask] = self.missing_val
        depth = depth.astype(np.int32)
        # Distribute depth values across RGB channels
        depth_image = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
        depth_image[..., 0] = depth & 0xFF  # Red channel
        depth_image[..., 1] = (depth >> 8) & 0xFF  # Green channel
        depth_image[..., 2] = (depth >> 16) & 0xFF  # Blue channel
        return depth_image

    def _write_distance_to_camera(self, data: dict, render_product_path: str, annotator: str):
        ## modified

        dist_to_cam_data = data[annotator]
        file_path = f"{render_product_path}distance_to_camera_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
        
        depth_image = self.split_png(dist_to_cam_data)
        # Save the image using cv2.imwrite
        cv2.imwrite(os.path.join(self._output_dir, file_path), depth_image)
        # buf = io.BytesIO()
        # np.save(buf, depth)
        # self._backend.write_blob(file_path, buf.getvalue())
        # depth_float = depth.astype(np.uint32)
        # cv2.imwrite(os.path.join(self._output_dir, file_path), depth, [cv2.IMWRITE_TIFF_COMPRESSION, 5])
        # cv2.imwrite(os.path.join(self._output_dir, file_path), depth)
        ## original
        # dist_to_cam_data = data[annotator]
        # file_path = f"{render_product_path}distance_to_camera_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        # buf = io.BytesIO()
        # np.save(buf, dist_to_cam_data)
        # # self._backend.write_blob(file_path, buf.getvalue())

    def _write_distance_to_image_plane(self, data: dict, render_product_path: str, annotator: str):
        dis_to_img_plane_data = data[annotator]
        file_path = f"{render_product_path}distance_to_image_plane_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
        # buf = io.BytesIO()
        # np.save(buf, dis_to_img_plane_data)
        # self._backend.write_blob(file_path, buf.getvalue())
        depth_image = self.split_png(dis_to_img_plane_data)
        # Save the image using cv2.imwrite
        cv2.imwrite(os.path.join(self._output_dir, file_path), depth_image)
        
    def _write_semantic_segmentation(self, data: dict, render_product_path: str, annotator: str):
        semantic_seg_data = data[annotator]["data"]
        height, width = semantic_seg_data.shape[:2]

        file_path = f"{render_product_path}semantic_segmentation_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
        if self.colorize_semantic_segmentation:
            semantic_seg_data = semantic_seg_data.view(np.uint8).reshape(height, width, -1)
            self._backend.write_image(file_path, semantic_seg_data)
        else:
            semantic_seg_data = semantic_seg_data.view(np.uint32).reshape(height, width)
            self._backend.write_image(file_path, semantic_seg_data)

        id_to_labels = data[annotator]["info"]["idToLabels"]
        file_path = f"{render_product_path}semantic_segmentation_labels_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        buf = io.BytesIO()
        buf.write(json.dumps({str(k): v for k, v in id_to_labels.items()}).encode())
        self._backend.write_blob(file_path, buf.getvalue())

    def _write_instance_id_segmentation(self, data: dict, render_product_path: str, annotator: str):
        instance_seg_data = data[annotator]["data"]
        height, width = instance_seg_data.shape[:2]

        file_path = f"{render_product_path}instance_id_segmentation_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
        if self.colorize_instance_id_segmentation:
            instance_seg_data = instance_seg_data.view(np.uint8).reshape(height, width, -1)
            self._backend.write_image(file_path, instance_seg_data)
        else:
            instance_seg_data = instance_seg_data.view(np.uint32).reshape(height, width)
            self._backend.write_image(file_path, instance_seg_data)

        id_to_labels = data[annotator]["info"]["idToLabels"]
        file_path = f"{render_product_path}instance_id_segmentation_mapping_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        buf = io.BytesIO()
        buf.write(json.dumps({str(k): v for k, v in id_to_labels.items()}).encode())
        self._backend.write_blob(file_path, buf.getvalue())

    def _write_instance_segmentation(self, data: dict, render_product_path: str, annotator: str):
        instance_seg_data = data[annotator]["data"]
        height, width = instance_seg_data.shape[:2]

        file_path = f"{render_product_path}instance_segmentation_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
        if self.colorize_instance_segmentation:
            instance_seg_data = instance_seg_data.view(np.uint8).reshape(height, width, -1)
            self._backend.write_image(file_path, instance_seg_data)
        else:
            instance_seg_data = instance_seg_data.view(np.uint32).reshape(height, width)
            self._backend.write_image(file_path, instance_seg_data)

        id_to_labels = data[annotator]["info"]["idToLabels"]
        file_path = f"{render_product_path}instance_segmentation_mapping_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        buf = io.BytesIO()
        buf.write(json.dumps({str(k): v for k, v in id_to_labels.items()}).encode())
        self._backend.write_blob(file_path, buf.getvalue())

        id_to_semantics = data[annotator]["info"]["idToSemantics"]
        file_path = f"{render_product_path}instance_segmentation_semantics_mapping_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        buf = io.BytesIO()
        buf.write(json.dumps({str(k): v for k, v in id_to_semantics.items()}).encode())
        self._backend.write_blob(file_path, buf.getvalue())

    def _write_motion_vectors(self, data: dict, render_product_path: str, annotator: str):
        motion_vec_data = data[annotator]
        file_path = f"{render_product_path}motion_vectors_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        buf = io.BytesIO()
        np.save(buf, motion_vec_data)
        self._backend.write_blob(file_path, buf.getvalue())

    def _write_occlusion(self, data: dict, render_product_path: str, annotator: str):
        occlusion_data = data[annotator]
        file_path = f"{render_product_path}occlusion_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        buf = io.BytesIO()
        np.save(buf, occlusion_data)
        self._backend.write_blob(file_path, buf.getvalue())

    def _write_bounding_box_data(self, data: dict, bbox_type: str, render_product_path: str, annotator: str):
        bbox_data = data[annotator]["data"]
        id_to_labels = data[annotator]["info"]["idToLabels"]

        file_path = f"{render_product_path}bounding_box_{bbox_type}_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        buf = io.BytesIO()
        np.save(buf, bbox_data)
        self._backend.write_blob(file_path, buf.getvalue())

        labels_file_path = f"{render_product_path}bounding_box_{bbox_type}_labels_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        buf = io.BytesIO()
        buf.write(json.dumps(id_to_labels).encode())
        self._backend.write_blob(labels_file_path, buf.getvalue())

    def _write_camera_params(self, data: dict, render_product_path: str, annotator: str):
        camera_data = data[annotator]
        serializable_data = {}

        for key, val in camera_data.items():
            if isinstance(val, np.ndarray):
                serializable_data[key] = val.tolist()
            else:
                serializable_data[key] = val

        file_path = f"{render_product_path}camera_params_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        buf = io.BytesIO()
        buf.write(json.dumps(serializable_data).encode())
        self._backend.write_blob(file_path, buf.getvalue())

    def _write_pointcloud(self, data: dict, render_product_path: str, annotator: str):
        pointcloud_data = data[annotator]["data"]
        pointcloud_rgb = data[annotator]["info"]["pointRgb"].reshape(-1, 4)
        pointcloud_normals = data[annotator]["info"]["pointNormals"].reshape(-1, 4)
        pointcloud_semantic = data[annotator]["info"]["pointSemantic"]

        file_path = f"{render_product_path}pointcloud_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        buf = io.BytesIO()
        np.save(buf, pointcloud_data)
        self.backend.write_blob(file_path, buf.getvalue())

        rgb_file_path = f"{render_product_path}pointcloud_rgb_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        buf = io.BytesIO()
        np.save(buf, pointcloud_rgb)
        self.backend.write_blob(rgb_file_path, buf.getvalue())

        normals_file_path = f"{render_product_path}pointcloud_normals_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        buf = io.BytesIO()
        np.save(buf, pointcloud_normals)
        self.backend.write_blob(normals_file_path, buf.getvalue())

        semancit_file_path = f"{render_product_path}pointcloud_semantic_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        buf = io.BytesIO()
        np.save(buf, pointcloud_semantic)
        self.backend.write_blob(semancit_file_path, buf.getvalue())

    def _write_skeleton(self, data: dict, render_product_path: str, annotator: str):
        skeleton = json.loads(data[annotator]["skeletonData"])

        file_path = f"{render_product_path}skeleton_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"

        serializable_data = {f"skeleton_{idx}": skel for idx, skel in enumerate(skeleton)}

        buf = io.BytesIO()
        buf.write(json.dumps(serializable_data).encode())
        self.backend.write_blob(file_path, buf.getvalue())


WriterRegistry.register(BasicWriter)
