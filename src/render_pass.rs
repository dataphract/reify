use std::{collections::HashMap, ffi::CString};

use ash::vk;

use crate::{
    graph::{
        node::{NodeContext, NodeOutputs, OutputImage, OwnedNodeOutputs},
        BoxNode, GraphImage, GraphKey, Node,
    },
    Device, GraphEditor,
};

pub trait RenderPass {
    fn color_attachments(&self) -> &[ColorAttachmentInfo] {
        &[]
    }

    fn debug_label(&self) -> CString {
        c"[unlabeled render pass node]".into()
    }
}

pub struct RenderPassBuilder<'graph, R> {
    graph: &'graph mut GraphEditor,
    label: String,

    pass: R,
    slots: RenderPassSlots,

    pipelines: Vec<RenderPassGraphicsPipeline>,
}

impl<'graph, R> RenderPassBuilder<'graph, R>
where
    R: RenderPass + 'static,
{
    pub(crate) fn new(
        graph: &'graph mut GraphEditor,
        label: String,
        pass: R,
    ) -> RenderPassBuilder<'graph, R> {
        RenderPassBuilder {
            graph,
            label,
            pass,
            slots: RenderPassSlots::default(),
            pipelines: Vec::new(),
        }
    }

    pub fn build(self) -> GraphKey {
        // TODO(dp): update capacity with other attachments
        let mut outputs = OwnedNodeOutputs {
            images: Vec::with_capacity(self.pass.color_attachments().len()),
        };

        for att in self.slots.color_attachments.values() {
            outputs.images.push(OutputImage {
                resource: att.produce,
                consumed: att.consume,
                stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            });
        }

        self.graph.add_node(
            self.label,
            RenderPassNode {
                pass: self.pass,
                _slots: self.slots,
                outputs,
                pipelines: self.pipelines,
            },
        )
    }

    #[must_use]
    pub fn set_color_attachment(
        mut self,
        label: String,
        image: GraphImage,
        consume: Option<GraphImage>,
    ) -> Self {
        if !self
            .pass
            .color_attachments()
            .iter()
            .enumerate()
            .any(|(_, slot)| slot.label == label)
        {
            panic!("No slot with the label `{label}`");
        };

        assert!(!self.slots.color_attachments.contains_key(&label));
        self.slots.color_attachments.insert(
            label,
            RenderPassOutputSlot {
                consume,
                produce: image,
            },
        );

        self
    }

    #[must_use]
    pub fn add_graphics_pipeline<P>(mut self, device: &Device, pipeline: P) -> Self
    where
        P: GraphicsPipeline + 'static,
    {
        let storage = create_graphics_pipeline_storage(device, &pipeline);
        self.pipelines.push(RenderPassGraphicsPipeline {
            storage,
            object: Box::new(pipeline),
        });
        self
    }
}

#[derive(Default)]
struct RenderPassSlots {
    color_attachments: HashMap<String, RenderPassOutputSlot>,
}

struct RenderPassOutputSlot {
    pub(crate) consume: Option<GraphImage>,
    pub(crate) produce: GraphImage,
}

struct RenderPassNode<R> {
    pass: R,

    _slots: RenderPassSlots,
    outputs: OwnedNodeOutputs,

    pipelines: Vec<RenderPassGraphicsPipeline>,
}

unsafe impl<R: RenderPass + 'static> Node for RenderPassNode<R> {
    #[inline]
    fn outputs(&self) -> NodeOutputs {
        self.outputs.as_node_outputs()
    }

    #[inline]
    fn debug_label(&self) -> CString {
        self.pass.debug_label()
    }

    unsafe fn execute(&self, cx: &mut NodeContext) {
        // TODO(dp): make method on ColorAttachmentInfo?
        let color_attachment_info = |att: &ColorAttachmentInfo| {
            let load_op = vk::AttachmentLoadOp::from(att.load_op);
            let clear_color = if let LoadOp::Clear(c) = att.load_op {
                c.into()
            } else {
                vk::ClearColorValue { float32: [0.0; 4] }
            };

            let slot = self
                ._slots
                .color_attachments
                .get(&att.label)
                .expect("no such attachment");

            vk::RenderingAttachmentInfo::default()
                .image_view(cx.default_image_view(slot.produce))
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .resolve_mode(vk::ResolveModeFlags::NONE)
                .load_op(load_op)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: clear_color })
        };

        let color_attachments = self
            .pass
            .color_attachments()
            .iter()
            .map(color_attachment_info)
            .collect::<Vec<_>>();

        let color_att_0 = self
            ._slots
            .color_attachments
            .get(&self.pass.color_attachments()[0].label)
            .expect("no color attachment");

        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: *cx.image_info(color_att_0.produce).extent.as_2d().unwrap(),
        };

        let rendering_info = vk::RenderingInfo::default()
            .flags(vk::RenderingFlags::empty())
            .render_area(render_area)
            .layer_count(1)
            .view_mask(0)
            .color_attachments(&color_attachments);

        // TODO
        // if let Some(depth) = depth_attachment {
        //     render_info = render_info.depth_attachment(&depth);
        // }
        // if let Some(stencil) = stencil_attachment {
        //     render_info = render_info.stencil_attachment(&stencil);
        // }

        let device = cx.device().clone();
        let cmdbuf = cx.command_buffer();
        unsafe {
            device.cmd_begin_rendering(cmdbuf, &rendering_info);
        }

        // TODO run pipelines
        for pipeline in &self.pipelines {
            unsafe {
                cx.enter_debug_span(&pipeline.object.debug_label(), [0.6, 1.0, 0.6, 1.0]);

                device.cmd_bind_pipeline(
                    cmdbuf,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.storage.handle,
                );
                device.cmd_set_viewport(
                    cmdbuf,
                    0,
                    &[vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: render_area.extent.width as f32,
                        height: render_area.extent.height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );
                device.cmd_set_scissor(cmdbuf, 0, &[render_area]);

                let mut pipe_instance = GraphicsPipelineInstance { cx };
                pipeline.object.execute(&mut pipe_instance);

                cx.exit_debug_span();
            }
        }

        unsafe {
            cx.device().cmd_end_rendering(cx.command_buffer());
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum ClearColor {
    Float([f32; 4]),
    SInt([i32; 4]),
    UInt([u32; 4]),
}

impl From<ClearColor> for vk::ClearColorValue {
    fn from(value: ClearColor) -> Self {
        match value {
            ClearColor::Float(f) => vk::ClearColorValue { float32: f },
            ClearColor::SInt(s) => vk::ClearColorValue { int32: s },
            ClearColor::UInt(u) => vk::ClearColorValue { uint32: u },
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum LoadOp<T> {
    Load,
    Clear(T),
    DontCare,
}

impl<T> From<LoadOp<T>> for vk::AttachmentLoadOp {
    fn from(value: LoadOp<T>) -> Self {
        match value {
            LoadOp::Load => vk::AttachmentLoadOp::LOAD,
            LoadOp::Clear(_) => vk::AttachmentLoadOp::CLEAR,
            LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
        }
    }
}

pub struct OutputAttachmentInfo<T> {
    /// The label of the output attachment.
    ///
    /// This label is used to uniquely identify the output attachment among all the node's outputs.
    pub label: String,

    /// The image format of the output attachment.
    ///
    /// If `None`, the format is inferred.
    pub format: Option<vk::Format>,

    /// The load operation used to load the attachment.
    pub load_op: LoadOp<T>,
}

pub type ColorAttachmentInfo = OutputAttachmentInfo<ClearColor>;

pub trait GraphicsPipeline {
    fn vertex_info(&self) -> GraphicsPipelineVertexInfo;
    fn primitive_info(&self) -> GraphicsPipelinePrimitiveInfo;
    fn fragment_info(&self) -> GraphicsPipelineFragmentInfo;
    fn attachment_info(&self) -> GraphicsPipelineAttachmentInfo;

    fn debug_label(&self) -> CString {
        c"[unlabeled graphics pipeline]".into()
    }

    fn execute(&self, pipe: &mut GraphicsPipelineInstance);
}

pub struct GraphicsPipelineVertexInfo {
    pub shader_spv: Vec<u32>,
    pub entry: CString,
}

pub struct GraphicsPipelinePrimitiveInfo {
    pub topology: vk::PrimitiveTopology,
    pub primitive_restart_enable: bool,
    pub polygon_mode: vk::PolygonMode,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
}

pub struct GraphicsPipelineFragmentInfo {
    pub shader_spv: Vec<u32>,
    pub entry: CString,
}

pub struct GraphicsPipelineAttachmentInfo {
    pub color: Vec<vk::Format>,
    pub depth: vk::Format,
    pub stencil: vk::Format,
}

pub struct GraphicsPipelineStorage {
    handle: vk::Pipeline,
    _layout: vk::PipelineLayout,
    _vert_module: vk::ShaderModule,
    _frag_module: vk::ShaderModule,
}

pub struct RenderPassGraphicsPipeline {
    storage: GraphicsPipelineStorage,
    object: Box<dyn GraphicsPipeline>,
}

pub struct GraphicsPipelineInstance<'pipe, 'node, 'frame> {
    cx: &'pipe mut NodeContext<'node, 'frame>,
}

impl<'pipe, 'node, 'frame> GraphicsPipelineInstance<'pipe, 'node, 'frame> {
    pub fn draw(&mut self, vertex_count: u32, first_vertex: u32) {
        let device = self.cx.device();
        let cmdbuf = self.cx.command_buffer();

        unsafe {
            device.cmd_draw(cmdbuf, vertex_count, 1, first_vertex, 0);
        }
    }
}

fn create_graphics_pipeline_storage<P>(device: &Device, pipeline: &P) -> GraphicsPipelineStorage
where
    P: GraphicsPipeline,
{
    let vertex_info = pipeline.vertex_info();
    let primitive_info = pipeline.primitive_info();
    let fragment_info = pipeline.fragment_info();
    let attachment_info = pipeline.attachment_info();

    let vert_module = unsafe {
        device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::default()
                    .flags(vk::ShaderModuleCreateFlags::empty())
                    .code(&vertex_info.shader_spv),
            )
            .unwrap()
    };

    let frag_module = unsafe {
        device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::default()
                    .flags(vk::ShaderModuleCreateFlags::empty())
                    .code(&fragment_info.shader_spv),
            )
            .unwrap()
    };

    let stages = &[
        vk::PipelineShaderStageCreateInfo::default()
            .flags(vk::PipelineShaderStageCreateFlags::empty())
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(&vertex_info.entry),
        vk::PipelineShaderStageCreateInfo::default()
            .flags(vk::PipelineShaderStageCreateFlags::empty())
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(&fragment_info.entry),
    ];

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .flags(vk::PipelineVertexInputStateCreateFlags::empty())
        // TODO
        .vertex_binding_descriptions(&[])
        // TODO
        .vertex_attribute_descriptions(&[]);

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .flags(vk::PipelineInputAssemblyStateCreateFlags::empty())
        .topology(primitive_info.topology)
        .primitive_restart_enable(primitive_info.primitive_restart_enable);

    // Viewport state is dynamic, so no viewport or scissor info.
    //
    // TODO(dp): support multiple viewports
    let viewports = &[vk::Viewport::default()];
    let scissors = &[vk::Rect2D::default()];
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .flags(vk::PipelineViewportStateCreateFlags::empty())
        .viewports(viewports)
        .scissors(scissors);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(primitive_info.polygon_mode)
        .cull_mode(primitive_info.cull_mode)
        .front_face(primitive_info.front_face)
        // TODO(dp): support depth bias
        .depth_bias_enable(false)
        .depth_bias_constant_factor(0.0)
        .depth_bias_clamp(0.0)
        .depth_bias_slope_factor(1.0)
        .line_width(1.0);

    // TODO(dp): support multisampling
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
        .flags(vk::PipelineMultisampleStateCreateFlags::default())
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .sample_shading_enable(false)
        .min_sample_shading(0.0)
        .sample_mask(&[])
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false);

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::ALWAYS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false)
        .front(vk::StencilOpState {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::NEVER,
            compare_mask: 0,
            write_mask: 0,
            reference: 0,
        })
        .back(vk::StencilOpState {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::NEVER,
            compare_mask: 0,
            write_mask: 0,
            reference: 0,
        });

    let color_blend_attachments = &[vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .flags(vk::PipelineColorBlendStateCreateFlags::empty())
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::NO_OP)
        .attachments(color_blend_attachments)
        .blend_constants([1.0; 4]);

    let dynamic_states = &[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
        .flags(vk::PipelineDynamicStateCreateFlags::empty())
        .dynamic_states(dynamic_states);

    let mut pipeline_rendering_create_info = vk::PipelineRenderingCreateInfo::default()
        .view_mask(0)
        .color_attachment_formats(&attachment_info.color)
        .depth_attachment_format(attachment_info.depth)
        .stencil_attachment_format(attachment_info.stencil);

    let layout_create_info = vk::PipelineLayoutCreateInfo::default()
        .flags(vk::PipelineLayoutCreateFlags::empty())
        .set_layouts(&[])
        .push_constant_ranges(&[]);

    let layout = unsafe { device.create_pipeline_layout(&layout_create_info).unwrap() };

    let create_infos = &[vk::GraphicsPipelineCreateInfo::default()
        .flags(vk::PipelineCreateFlags::empty())
        .stages(stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        // No tessellation state.
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .dynamic_state(&dynamic_state)
        .layout(layout)
        .render_pass(vk::RenderPass::null())
        .subpass(0)
        .base_pipeline_handle(vk::Pipeline::null())
        .base_pipeline_index(-1)
        .push_next(&mut pipeline_rendering_create_info)];

    let pipeline = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), create_infos)
            .unwrap()[0]
    };

    GraphicsPipelineStorage {
        handle: pipeline,
        _layout: layout,
        _vert_module: vert_module,
        _frag_module: frag_module,
    }
}
