use std::ffi::CString;

use ash::vk;

use crate::{
    graph::{
        node::{
            InputImage, NodeContext, NodeInputs, NodeOutputs, OutputImage, OwnedNodeInputs,
            OwnedNodeOutputs,
        },
        GraphImage, Node,
    },
    image::ImageExtent,
};

pub struct BlitNode {
    src: GraphImage,
    dst: GraphImage,
    dst_consume: Option<GraphImage>,

    inputs: OwnedNodeInputs,
    outputs: OwnedNodeOutputs,
}

impl BlitNode {
    pub fn new(src: GraphImage, dst: GraphImage, consume: Option<GraphImage>) -> BlitNode {
        let inputs = OwnedNodeInputs {
            images: vec![input_image(src)],
        };

        let outputs = OwnedNodeOutputs {
            images: vec![output_image(dst, consume)],
        };

        BlitNode {
            src,
            dst,
            dst_consume: consume,
            inputs,
            outputs,
        }
    }
}

fn input_image(img: GraphImage) -> InputImage {
    InputImage {
        resource: img,
        stage_mask: vk::PipelineStageFlags2::BLIT,
        access_mask: vk::AccessFlags2::TRANSFER_READ,
        layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        usage: vk::ImageUsageFlags::TRANSFER_SRC,
    }
}

fn output_image(img: GraphImage, consumed: Option<GraphImage>) -> OutputImage {
    OutputImage {
        resource: img,
        consumed,
        stage_mask: vk::PipelineStageFlags2::BLIT,
        access_mask: vk::AccessFlags2::TRANSFER_WRITE,
        layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        usage: vk::ImageUsageFlags::TRANSFER_DST,
    }
}

unsafe impl Node for BlitNode {
    fn inputs(&self) -> NodeInputs {
        self.inputs.as_node_inputs()
    }

    fn outputs(&self) -> NodeOutputs {
        self.outputs.as_node_outputs()
    }

    fn debug_label(&self) -> CString {
        c"blit node".into()
    }

    unsafe fn execute(&self, cx: &mut NodeContext) {
        let device = cx.device();
        let cmdbuf = cx.command_buffer();

        let full_subresource = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1);

        let offset_zero = vk::Offset3D::default();

        let src_info = cx.image_info(self.src);
        let src_end = match src_info.extent {
            ImageExtent::D2(x2d) => vk::Offset3D {
                x: x2d.width as i32,
                y: x2d.height as i32,
                z: 1,
            },
        };

        let dst_info = cx.image_info(self.dst);
        let dst_end = match dst_info.extent {
            ImageExtent::D2(x2d) => vk::Offset3D {
                x: x2d.width as i32,
                y: x2d.height as i32,
                z: 1,
            },
        };

        let region = vk::ImageBlit2::default()
            .src_subresource(full_subresource)
            .src_offsets([offset_zero, src_end])
            .dst_subresource(full_subresource)
            .dst_offsets([offset_zero, dst_end]);

        let regions = &[region];

        let info = vk::BlitImageInfo2::default()
            .src_image(cx.image(self.src))
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_image(cx.image(self.dst))
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .regions(regions);

        unsafe { device.cmd_blit_image2(cmdbuf, &info) }
    }
}
