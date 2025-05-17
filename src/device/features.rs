bitflags::bitflags! {
    pub struct PhysicalDeviceFeaturesFlags: u64 {
        const ROBUST_BUFFER_ACCESS                         = 1 <<  0;
        const FULL_DRAW_INDEX_UINT32                       = 1 <<  1;
        const IMAGE_CUBE_ARRAY                             = 1 <<  2;
        const INDEPENDENT_BLEND                            = 1 <<  3;
        const GEOMETRY_SHADER                              = 1 <<  4;
        const TESSELLATION_SHADER                          = 1 <<  5;
        const SAMPLE_RATE_SHADING                          = 1 <<  6;
        const DUAL_SRC_BLEND                               = 1 <<  7;
        const LOGIC_OP                                     = 1 <<  8;
        const MULTI_DRAW_INDIRECT                          = 1 <<  9;
        const DRAW_INDIRECT_FIRST_INSTANCE                 = 1 << 10;
        const DEPTH_CLAMP                                  = 1 << 11;
        const DEPTH_BIAS_CLAMP                             = 1 << 12;
        const FILL_MODE_NON_SOLID                          = 1 << 13;
        const DEPTH_BOUNDS                                 = 1 << 14;
        const WIDE_LINES                                   = 1 << 15;
        const LARGE_POINTS                                 = 1 << 16;
        const ALPHA_TO_ONE                                 = 1 << 17;
        const MULTI_VIEWPORT                               = 1 << 18;
        const SAMPLER_ANISOTROPY                           = 1 << 19;
        const TEXTURE_COMPRESSION_ETC2                     = 1 << 20;
        const TEXTURE_COMPRESSION_ASTC_LDR                 = 1 << 21;
        const TEXTURE_COMPRESSION_BC                       = 1 << 22;
        const OCCLUSION_QUERY_PRECISE                      = 1 << 23;
        const PIPELINE_STATISTICS_QUERY                    = 1 << 24;
        const VERTEX_PIPELINE_STORES_AND_ATOMICS           = 1 << 25;
        const FRAGMENT_STORES_AND_ATOMICS                  = 1 << 26;
        const SHADER_TESSELLATION_AND_GEOMETRY_POINT_SIZE  = 1 << 27;
        const SHADER_IMAGE_GATHER_EXTENDED                 = 1 << 28;
        const SHADER_STORAGE_IMAGE_EXTENDED_FORMATS        = 1 << 29;
        const SHADER_STORAGE_IMAGE_MULTISAMPLE             = 1 << 30;
        const SHADER_STORAGE_IMAGE_READ_WITHOUT_FORMAT     = 1 << 31;
        const SHADER_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT    = 1 << 32;
        const SHADER_UNIFORM_BUFFER_ARRAY_DYNAMIC_INDEXING = 1 << 33;
        const SHADER_SAMPLED_IMAGE_ARRAY_DYNAMIC_INDEXING  = 1 << 34;
        const SHADER_STORAGE_BUFFER_ARRAY_DYNAMIC_INDEXING = 1 << 35;
        const SHADER_STORAGE_IMAGE_ARRAY_DYNAMIC_INDEXING  = 1 << 36;
        const SHADER_CLIP_DISTANCE                         = 1 << 37;
        const SHADER_CULL_DISTANCE                         = 1 << 38;
        const SHADER_FLOAT64                               = 1 << 39;
        const SHADER_INT64                                 = 1 << 40;
        const SHADER_INT16                                 = 1 << 41;
        const SHADER_RESOURCE_RESIDENCY                    = 1 << 42;
        const SHADER_RESOURCE_MIN_LOD                      = 1 << 43;
        const SPARSE_BINDING                               = 1 << 44;
        const SPARSE_RESIDENCY_BUFFER                      = 1 << 45;
        const SPARSE_RESIDENCY_IMAGE_2D                    = 1 << 46;
        const SPARSE_RESIDENCY_IMAGE_3D                    = 1 << 47;
        const SPARSE_RESIDENCY_2_SAMPLES                    = 1 << 48;
        const SPARSE_RESIDENCY_4_SAMPLES                    = 1 << 49;
        const SPARSE_RESIDENCY_8_SAMPLES                    = 1 << 50;
        const SPARSE_RESIDENCY_16_SAMPLES                   = 1 << 51;
        const SPARSE_RESIDENCY_ALIASED                     = 1 << 52;
        const VARIABLE_MULTISAMPLE_RATE                    = 1 << 53;
        const INHERITED_QUERIES                            = 1 << 54;
    }
}

impl From<&ash::vk::PhysicalDeviceFeatures> for PhysicalDeviceFeaturesFlags {
    fn from(features: &ash::vk::PhysicalDeviceFeatures) -> Self {
        let mut flags = Self::empty();
        if features.robust_buffer_access == ash::vk::TRUE {
            flags |= Self::ROBUST_BUFFER_ACCESS;
        }
        if features.full_draw_index_uint32 == ash::vk::TRUE {
            flags |= Self::FULL_DRAW_INDEX_UINT32;
        }
        if features.image_cube_array == ash::vk::TRUE {
            flags |= Self::IMAGE_CUBE_ARRAY;
        }
        if features.independent_blend == ash::vk::TRUE {
            flags |= Self::INDEPENDENT_BLEND;
        }
        if features.geometry_shader == ash::vk::TRUE {
            flags |= Self::GEOMETRY_SHADER;
        }
        if features.tessellation_shader == ash::vk::TRUE {
            flags |= Self::TESSELLATION_SHADER;
        }
        if features.sample_rate_shading == ash::vk::TRUE {
            flags |= Self::SAMPLE_RATE_SHADING;
        }
        if features.dual_src_blend == ash::vk::TRUE {
            flags |= Self::DUAL_SRC_BLEND;
        }
        if features.logic_op == ash::vk::TRUE {
            flags |= Self::LOGIC_OP;
        }
        if features.multi_draw_indirect == ash::vk::TRUE {
            flags |= Self::MULTI_DRAW_INDIRECT;
        }
        if features.draw_indirect_first_instance == ash::vk::TRUE {
            flags |= Self::DRAW_INDIRECT_FIRST_INSTANCE;
        }
        if features.depth_clamp == ash::vk::TRUE {
            flags |= Self::DEPTH_CLAMP;
        }
        if features.depth_bias_clamp == ash::vk::TRUE {
            flags |= Self::DEPTH_BIAS_CLAMP;
        }
        if features.fill_mode_non_solid == ash::vk::TRUE {
            flags |= Self::FILL_MODE_NON_SOLID;
        }
        if features.depth_bounds == ash::vk::TRUE {
            flags |= Self::DEPTH_BOUNDS;
        }
        if features.wide_lines == ash::vk::TRUE {
            flags |= Self::WIDE_LINES;
        }
        if features.large_points == ash::vk::TRUE {
            flags |= Self::LARGE_POINTS;
        }
        if features.alpha_to_one == ash::vk::TRUE {
            flags |= Self::ALPHA_TO_ONE;
        }
        if features.multi_viewport == ash::vk::TRUE {
            flags |= Self::MULTI_VIEWPORT;
        }
        if features.sampler_anisotropy == ash::vk::TRUE {
            flags |= Self::SAMPLER_ANISOTROPY;
        }
        if features.texture_compression_etc2 == ash::vk::TRUE {
            flags |= Self::TEXTURE_COMPRESSION_ETC2;
        }
        if features.texture_compression_astc_ldr == ash::vk::TRUE {
            flags |= Self::TEXTURE_COMPRESSION_ASTC_LDR;
        }
        if features.texture_compression_bc == ash::vk::TRUE {
            flags |= Self::TEXTURE_COMPRESSION_BC;
        }
        if features.occlusion_query_precise == ash::vk::TRUE {
            flags |= Self::OCCLUSION_QUERY_PRECISE;
        }
        if features.pipeline_statistics_query == ash::vk::TRUE {
            flags |= Self::PIPELINE_STATISTICS_QUERY;
        }
        if features.vertex_pipeline_stores_and_atomics == ash::vk::TRUE {
            flags |= Self::VERTEX_PIPELINE_STORES_AND_ATOMICS;
        }
        if features.fragment_stores_and_atomics == ash::vk::TRUE {
            flags |= Self::FRAGMENT_STORES_AND_ATOMICS;
        }
        if features.shader_tessellation_and_geometry_point_size == ash::vk::TRUE {
            flags |= Self::SHADER_TESSELLATION_AND_GEOMETRY_POINT_SIZE;
        }
        if features.shader_image_gather_extended == ash::vk::TRUE {
            flags |= Self::SHADER_IMAGE_GATHER_EXTENDED;
        }
        if features.shader_storage_image_extended_formats == ash::vk::TRUE {
            flags |= Self::SHADER_STORAGE_IMAGE_EXTENDED_FORMATS;
        }
        if features.shader_storage_image_multisample == ash::vk::TRUE {
            flags |= Self::SHADER_STORAGE_IMAGE_MULTISAMPLE;
        }
        if features.shader_storage_image_read_without_format == ash::vk::TRUE {
            flags |= Self::SHADER_STORAGE_IMAGE_READ_WITHOUT_FORMAT;
        }
        if features.shader_storage_image_write_without_format == ash::vk::TRUE {
            flags |= Self::SHADER_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT;
        }
        if features.shader_uniform_buffer_array_dynamic_indexing == ash::vk::TRUE {
            flags |= Self::SHADER_UNIFORM_BUFFER_ARRAY_DYNAMIC_INDEXING;
        }
        if features.shader_sampled_image_array_dynamic_indexing == ash::vk::TRUE {
            flags |= Self::SHADER_SAMPLED_IMAGE_ARRAY_DYNAMIC_INDEXING;
        }
        if features.shader_storage_buffer_array_dynamic_indexing == ash::vk::TRUE {
            flags |= Self::SHADER_STORAGE_BUFFER_ARRAY_DYNAMIC_INDEXING;
        }
        if features.shader_storage_image_array_dynamic_indexing == ash::vk::TRUE {
            flags |= Self::SHADER_STORAGE_IMAGE_ARRAY_DYNAMIC_INDEXING;
        }
        if features.shader_clip_distance == ash::vk::TRUE {
            flags |= Self::SHADER_CLIP_DISTANCE;
        }
        if features.shader_cull_distance == ash::vk::TRUE {
            flags |= Self::SHADER_CULL_DISTANCE;
        }
        if features.shader_float64 == ash::vk::TRUE {
            flags |= Self::SHADER_FLOAT64;
        }
        if features.shader_int64 == ash::vk::TRUE {
            flags |= Self::SHADER_INT64;
        }
        if features.shader_int16 == ash::vk::TRUE {
            flags |= Self::SHADER_INT16;
        }
        if features.shader_resource_residency == ash::vk::TRUE {
            flags |= Self::SHADER_RESOURCE_RESIDENCY;
        }
        if features.shader_resource_min_lod == ash::vk::TRUE {
            flags |= Self::SHADER_RESOURCE_MIN_LOD;
        }
        if features.sparse_binding == ash::vk::TRUE {
            flags |= Self::SPARSE_BINDING;
        }
        if features.sparse_residency_buffer == ash::vk::TRUE {
            flags |= Self::SPARSE_RESIDENCY_BUFFER;
        }
        if features.sparse_residency_image2_d == ash::vk::TRUE {
            flags |= Self::SPARSE_RESIDENCY_IMAGE_2D;
        }
        if features.sparse_residency_image3_d == ash::vk::TRUE {
            flags |= Self::SPARSE_RESIDENCY_IMAGE_3D;
        }
        if features.sparse_residency2_samples == ash::vk::TRUE {
            flags |= Self::SPARSE_RESIDENCY_2_SAMPLES;
        }
        if features.sparse_residency4_samples == ash::vk::TRUE {
            flags |= Self::SPARSE_RESIDENCY_4_SAMPLES;
        }
        if features.sparse_residency8_samples == ash::vk::TRUE {
            flags |= Self::SPARSE_RESIDENCY_8_SAMPLES;
        }
        if features.sparse_residency16_samples == ash::vk::TRUE {
            flags |= Self::SPARSE_RESIDENCY_16_SAMPLES;
        }
        if features.sparse_residency_aliased == ash::vk::TRUE {
            flags |= Self::SPARSE_RESIDENCY_ALIASED;
        }
        if features.variable_multisample_rate == ash::vk::TRUE {
            flags |= Self::VARIABLE_MULTISAMPLE_RATE;
        }
        if features.inherited_queries == ash::vk::TRUE {
            flags |= Self::INHERITED_QUERIES;
        }
        flags
    }
}
