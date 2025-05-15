// Delta-compressed draw command buffer.
//
// Based on design described in
//
//   Sebastian Aaltonen. 2023. HypeHype Mobile Rendering Architecture.
//   Presentation at ACM SIGGRAPH 2023, Advances in Real-Time Rendering in Games, Los Angeles, CA, USA.
//   https://advances.realtimerendering.com/s2023/AaltonenHypeHypeAdvances2023.pdf

// TODO: make use of VK_EXT_descriptor_buffer on devices that have it.
// Coverage is fairly low as of 2025, so this is a lower-priority optimization.
// See https://vulkan.gpuinfo.org/displayextensiondetail.php?extension=VK_EXT_descriptor_buffer

// Incomplete list of Vulkan commands affecting draw commands within a render pass, as of 2025:
//
// Name (without extension suffix)     | Version | Extension
// ====================================|=========|====================================================
// vkCmdBindPipeline                   | 1.0     |
// vkCmdBindShaders                    |         | VK_EXT_shader_object
// ------------------------------------|---------|----------------------------------------------------
// vkCmdBindDescriptorSets             | 1.0     |
// vkCmdBindDescriptorSets2            | 1.0     | VK_KHR_maintenance6
// vkCmdPushDescriptorSet              | 1.4     | VK_KHR_push_descriptor
// vkCmdPushDescriptorSet2             | 1.4     | VK_KHR_mainenance6 AND VK_KHR_push_descriptor
// vkCmdPushDescriptorSetWithTemplate  | 1.4     | VK_KHR_descriptor_update_template
// vkCmdPushDescriptorSetWithTemplate2 | 1.4     | VK_KHR_push_descriptor AND VK_KHR_maintenance6
// vkCmdPushConstants                  | 1.0     |
// vkCmdPushConstants2                 | 1.4     | VK_KHR_maintenance6
// vkCmdBindDescriptorBuffers          | -/-     | VK_EXT_descriptor_buffer
// vkCmdSetDescriptorBufferOffsets     | -/-     | VK_EXT_descriptor_buffer
// vkCmdSetDescriptorBufferOffsets2    | -/-     | VK_EXT_descriptor_buffer AND VK_KHR_maintenance6
// ------------------------------------|---------|----------------------------------------------------
// vkCmdBindIndexBuffer                | 1.0     |
// vkCmdBindIndexBuffer2               | 1.4     | VK_KHR_maintenance5
// vkCmdBindVertexBuffers              | 1.0     |
// vkCmdBindVertexBuffers2             | 1.3     | VK_EXT_extended_dynamic_state, VK_EXT_shader_object
//

use bitflags::Flags;

// TODO: strongly typed fields
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Draw {
    // pipeline: u64,
    // bind_groups: [u64; 3],
    // index_buffer: u64,
    // vertex_buffers: [u64; 3],
    // index_base: u32,
    instance_base: u32,
    instance_count: u32,
    vertex_base: u32,
    vertex_count: u32,
}

impl Default for Draw {
    fn default() -> Self {
        Draw {
            // pipeline: 0,
            // bind_groups: [0; 3],
            // index_buffer: 0,
            // vertex_buffers: [0; 3],
            // index_base: 0,
            instance_base: 0,
            instance_count: 1,
            vertex_base: 0,
            vertex_count: 0,
        }
    }
}

bitflags::bitflags! {
    struct DrawFlags: u32 {
        // const PIPELINE       = 1 << 0;
        // const BIND_GROUP_0   = 1 << 1;
        // const BIND_GROUP_1   = 1 << 2;
        // const BIND_GROUP_2   = 1 << 3;
        const INSTANCE_BASE  = 1 << 28;
        const INSTANCE_COUNT = 1 << 29;
        const VERTEX_BASE    = 1 << 30;
        const VERTEX_COUNT   = 1 << 31;
    }
}

macro_rules! for_each_set_flag {
    (in ($flags:expr) {
        $($flag:ident => $work:expr),* $(,)?
    }) => {
        $(if $flags.contains(DrawFlags::$flag) { $work; })*
    }
}

pub struct CommandWriter<'a> {
    state: Draw,
    flags: DrawFlags,
    // TODO: may want to record over multiple fixed-size chunks per thread
    data: &'a mut Vec<u32>,
}

impl<'a> CommandWriter<'a> {
    pub(crate) fn new(data: &mut Vec<u32>) -> CommandWriter {
        CommandWriter {
            state: Draw::default(),
            flags: DrawFlags::all(),
            data,
        }
    }

    fn flush(&mut self) {
        self.data.push(self.flags.bits());

        for_each_set_flag!(in (self.flags) {
            // PIPELINE => self.write_u64(self.state.pipeline),
            // BIND_GROUP_0 => self.write_u64(self.state.bind_groups[0]),
            // BIND_GROUP_1 => self.write_u64(self.state.bind_groups[1]),
            // BIND_GROUP_2 => self.write_u64(self.state.bind_groups[2]),
            INSTANCE_COUNT => self.write_u32(self.state.instance_count),
            INSTANCE_BASE => self.write_u32(self.state.instance_base),
            VERTEX_COUNT => self.write_u32(self.state.vertex_count),
            VERTEX_BASE => self.write_u32(self.state.vertex_base),
        });

        self.flags.clear();
    }

    // #[inline]
    // fn write_u64(&mut self, value: u64) {
    //     self.data.push((value >> 32) as u32);
    //     self.data.push(value as u32);
    // }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.data.push(value);
    }

    pub fn draw(
        &mut self,
        instance_base: u32,
        instance_count: u32,
        vertex_base: u32,
        vertex_count: u32,
    ) {
        if instance_base != self.state.instance_base {
            self.state.instance_base = instance_base;
            self.flags |= DrawFlags::INSTANCE_BASE;
        }

        if instance_count != self.state.instance_count {
            self.state.instance_count = instance_count;
            self.flags |= DrawFlags::INSTANCE_COUNT;
        }

        if vertex_base != self.state.vertex_base {
            self.state.vertex_base = vertex_base;
            self.flags |= DrawFlags::VERTEX_BASE;
        }

        if vertex_count != self.state.vertex_count {
            self.state.vertex_count = vertex_count;
            self.flags |= DrawFlags::VERTEX_COUNT;
        }

        self.flush();
    }
}

pub(crate) struct CommandReader<'a> {
    state: Draw,

    inner: std::slice::Iter<'a, u32>,
}

impl<'a> CommandReader<'a> {
    fn new(data: &'a [u32]) -> Self {
        CommandReader {
            state: Draw::default(),
            inner: data.iter(),
        }
    }

    fn expect_u32(&mut self) -> Result<u32, crate::Error> {
        self.inner.next().copied().ok_or(crate::Error::Bug)
    }

    fn next_impl(&mut self) -> Result<Option<Draw>, crate::Error> {
        let Some(&flags) = self.inner.next() else {
            return Ok(None);
        };

        let flags = DrawFlags::from_bits_truncate(flags);

        for_each_set_flag!(in (flags) {
            INSTANCE_COUNT => self.state.instance_count = self.expect_u32()?,
            INSTANCE_BASE => self.state.instance_base = self.expect_u32()?,
            VERTEX_COUNT => self.state.vertex_count = self.expect_u32()?,
            VERTEX_BASE => self.state.vertex_base = self.expect_u32()?,
        });

        Ok(Some(self.state))
    }
}

impl<'a> Iterator for CommandReader<'a> {
    type Item = Result<Draw, crate::Error>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.next_impl().transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verifies that draw state parameters are written and read in the same order.
    #[test]
    fn read_write_order() {
        let mut buf = Vec::new();

        let mut w = CommandWriter::new(&mut buf);
        w.draw(0, 1, 2, 3);
        drop(w);

        let mut r = CommandReader::new(&buf);
        let draw = r
            .next()
            .expect("no draw command")
            .expect("error in draw stream");

        assert_eq!(
            draw,
            Draw {
                instance_base: 0,
                instance_count: 1,
                vertex_base: 2,
                vertex_count: 3,
            }
        );
    }
}
