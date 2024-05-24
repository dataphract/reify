use ash::vk;
use naga::front::glsl;

fn main() {
    examples::AppRunner::<TriangleApp>::new().run();
}

struct TriangleApp {
    _pipe_layout: vk::PipelineLayout,
    pipe: vk::Pipeline,
}

impl examples::App for TriangleApp {
    fn create_app(device: &reify2::Device, display_info: &reify2::DisplayInfo) -> TriangleApp {
        let vert_src = r#"
#version 460 core

void main() {
    vec2 verts[3] = vec2[](
        vec2(-0.4, -0.5),
        vec2(0.4, -0.5),
        vec2(0.0, 0.5)
    );

    gl_Position = vec4(verts[gl_VertexIndex], 0.0, 1.0);
}
"#;

        let frag_src = r#"
#version 460 core

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(0.0, 0.2, 1.0, 1.0);
}
"#;

        let mut glsl_front = glsl::Frontend::default();
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );

        let vert_module = compile_shader(
            device,
            &mut glsl_front,
            &mut validator,
            naga::ShaderStage::Vertex,
            vert_src,
        );
        let frag_module = compile_shader(
            device,
            &mut glsl_front,
            &mut validator,
            naga::ShaderStage::Fragment,
            frag_src,
        );

        let stages = &[
            vk::PipelineShaderStageCreateInfo::default()
                .flags(vk::PipelineShaderStageCreateFlags::empty())
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .flags(vk::PipelineShaderStageCreateFlags::empty())
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(c"main"),
        ];

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .flags(vk::PipelineVertexInputStateCreateFlags::empty())
            .vertex_binding_descriptions(&[])
            .vertex_attribute_descriptions(&[]);

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .flags(vk::PipelineInputAssemblyStateCreateFlags::empty())
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        // Viewport state is dynamic, so no viewport or scissor info.
        let viewports = &[vk::Viewport::default()];
        let scissors = &[vk::Rect2D::default()];
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .flags(vk::PipelineViewportStateCreateFlags::empty())
            .viewports(viewports)
            .scissors(scissors);

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(1.0)
            .line_width(1.0);

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

        let color_attachment_formats = &[display_info.surface_format.format];
        let mut pipeline_rendering_create_info = vk::PipelineRenderingCreateInfo::default()
            .view_mask(0)
            .color_attachment_formats(color_attachment_formats)
            .depth_attachment_format(vk::Format::UNDEFINED)
            .stencil_attachment_format(vk::Format::UNDEFINED);

        let layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .flags(vk::PipelineLayoutCreateFlags::empty())
            .set_layouts(&[])
            .push_constant_ranges(&[]);

        let pipe_layout = unsafe { device.create_pipeline_layout(&layout_create_info).unwrap() };

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
            .layout(pipe_layout)
            .render_pass(vk::RenderPass::null())
            .subpass(0)
            .base_pipeline_handle(vk::Pipeline::null())
            .base_pipeline_index(-1)
            .push_next(&mut pipeline_rendering_create_info)];

        let pipe = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), create_infos)
                .unwrap()[0]
        };

        TriangleApp {
            _pipe_layout: pipe_layout,
            pipe,
        }
    }

    fn render(&self, device: &reify2::Device, cx: &mut reify2::FrameContext) {
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(cx.swapchain_image().view())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_mode(vk::ResolveModeFlags::NONE)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            });
        let color_attachments = &[color_attachment];

        let display_info = cx.display_info();

        let rendering_info = vk::RenderingInfo::default()
            .flags(vk::RenderingFlags::empty())
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: cx.display_info().image_extent,
            })
            .layer_count(1)
            .view_mask(0)
            .color_attachments(color_attachments);

        unsafe {
            device.cmd_begin_rendering(cx.command_buffer(), &rendering_info);

            device.cmd_bind_pipeline(
                cx.command_buffer(),
                vk::PipelineBindPoint::GRAPHICS,
                self.pipe,
            );
            device.cmd_set_viewport(
                cx.command_buffer(),
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: display_info.image_extent.width as f32,
                    height: display_info.image_extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device.cmd_set_scissor(
                cx.command_buffer(),
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: display_info.image_extent,
                }],
            );
            device.cmd_draw(cx.command_buffer(), 3, 1, 0, 0);

            device.cmd_end_rendering(cx.command_buffer());
        }
    }
}

fn compile_shader(
    device: &reify2::Device,
    front: &mut naga::front::glsl::Frontend,
    validator: &mut naga::valid::Validator,
    stage: naga::ShaderStage,
    src: &str,
) -> vk::ShaderModule {
    let parsed = front
        .parse(
            &glsl::Options {
                stage,
                defines: Default::default(),
            },
            src,
        )
        .unwrap();

    let info = validator.validate(&parsed).unwrap();
    let spv_back_options = naga::back::spv::Options {
        lang_version: (1, 6),
        ..Default::default()
    };

    let spv = naga::back::spv::write_vec(
        &parsed,
        &info,
        &spv_back_options,
        Some(&naga::back::spv::PipelineOptions {
            shader_stage: stage,
            entry_point: "main".into(),
        }),
    )
    .unwrap();

    let create_info = vk::ShaderModuleCreateInfo::default()
        .flags(vk::ShaderModuleCreateFlags::empty())
        .code(&spv);

    unsafe { device.create_shader_module(&create_info).unwrap() }
}
