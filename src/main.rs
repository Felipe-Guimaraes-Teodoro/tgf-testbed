use std::{ptr, ffi::CString};

use tiny_game_framework::{
    bind_buffer, cstr, gen_attrib_pointers, gl::{types::*, *}, glam::{vec2, vec3, vec4, Vec2, Vec3, Vec4}, rand_betw, Cuboid, EventLoop, InstanceData, Mesh, Model, Renderer, Shader, Vertex
};

fn main() {
    let grass_shader_source_vs = r#"
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aInstancePos;

uniform mat4 view;
uniform mat4 proj;

out vec4 fColor;

mat4 scale(float c)
{
    return mat4(c, 0, 0, 0,
                0, c, 0, 0,
                0, 0, c, 0,
                0, 0, 0, 1);
}

mat4 rotation3dY(float angle) {
  float s = sin(angle);
  float c = cos(angle);

  return mat4(
    c,   0.0, -s,  0.0,
    0.0, 1.0, 0.0, 0.0,
    s,   0.0, c,   0.0,
    0.0, 0.0, 0.0, 1.0
  );
}

// @return Value of the noise, range: [0, 1]
float hash1D(float x)
{
    // based on: pcg by Mark Jarzynski: http://www.jcgt.org/published/0009/03/02/
    uint state = uint(x * 8192.0) * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) * (1.0 / float(0xffffffffu));;
}

uniform float time;

void main() {
    vec3 gOfs = aInstancePos;

    float angle = hash1D(gOfs.x + gOfs.y + gOfs.z) * 2.0 * 3.14159; // range [0, 2*PI]
    mat4 rotation = rotation3dY(angle);

    vec4 localPos = rotation * vec4(aPos, 1.0);

    mat4 scale = scale((hash1D(gOfs.x + gOfs.y + gOfs.z) + 0.5) * 0.5);

    // Generate wind parameters based on the position
    float windStrength = hash1D(gOfs.x * 0.1 + gOfs.y * 0.2 + gOfs.z * 0.3) * 2.0 + 2.0;
    vec3 windDirection = normalize(vec3(hash1D(gOfs.x * 0.2), 0.0, hash1D(gOfs.z * 0.3)));

    // Wind effect using sine wave and hash for variation
    float windOffsetX = sin(time + gOfs.x * 0.1 + hash1D(gOfs.y) * 2.0 * 3.14159) * 0.1;
    float windOffsetZ = cos(time + gOfs.z * 0.1 + hash1D(gOfs.y) * 2.0 * 3.14159) * 0.1;
    vec3 windEffect = vec3(windOffsetX, 0.0, windOffsetZ) * windStrength;

    // Combine wind direction with calculated wind effect
    vec3 k = windDirection * windEffect * localPos.y;

    gl_Position = proj * view * scale * vec4(localPos.xyz + gOfs + k, 1.0);
    fColor = vec4(localPos.y / 2.5 + 0.1, localPos.y, localPos.y / 2.2, 1.0);
}
"#;

    let grass_shader_source_fs = r#"
#version 330 core
out vec4 FragColor;

in vec4 fColor;

void main()
{
    FragColor = fColor;
}
"#;

    let resolution = vec2(800., 800.);
    let mut el = EventLoop::new(resolution.x as u32, resolution.y as u32);
    let mut renderer = Renderer::new();

    unsafe {
        Enable(DEPTH_TEST);
    }

    el.window.glfw.set_swap_interval(tiny_game_framework::glfw::SwapInterval::None);
    el.window.set_cursor_mode(tiny_game_framework::glfw::CursorMode::Disabled);
    
    let mut c = Cuboid::new(vec3(600., 600., 600.0), vec4(1.0, 1.0, 1.0, 1.0)).mesh();
    c.setup_mesh();
    renderer.add_mesh("c", c).unwrap();
    
    let grass_shader = Shader::new_pipeline(grass_shader_source_vs, grass_shader_source_fs);
    let mut grasses = vec![];
    
    for i in -25..0 {
        for j in -25..0 {
            let mut grid = grid(|x, y| { return rand_betw(-150.0, 150.0); }, 100, 100, 100.0);
            grid.setup_mesh();
            let grid_vertices = grid.vertices.clone();
            renderer.add_mesh(&format!("grid_{}_{}", i, j), grid.clone()).unwrap();
            grid.position = vec3(i as f32, 0.0, j as f32) * 17.0;
        
            let grass = populate_with_grass(&grid_vertices, grid.position);
            grasses.push(grass);
        }
    }

    while !el.window.should_close() {
        el.update();
        renderer.camera.mouse_callback(el.event_handler.mouse_pos.x, el.event_handler.mouse_pos.y, &el.window);
        renderer.camera.input(&el.window, &el.window.glfw);
        renderer.camera.update(renderer.camera.pos);

        let frame = el.ui.frame(&mut el.window);
        frame.text(format!("f: {:?}", 1.0/el.dt));

        unsafe {
            Clear(COLOR_BUFFER_BIT | DEPTH_BUFFER_BIT);
            ClearColor(0.1, 0.28, 0.4, 1.0);

            for grass in &grasses {
                if renderer.camera.pos.distance(grass.1.avg_pos / 4.0) > 50.0 { continue; } // don't even render it
                if renderer.camera.pos.distance(grass.1.avg_pos / 4.0) > 12.0 {
                    grass.1.draw(&grass_shader, &renderer, &el);
                } else {
                    grass.0.draw(&grass_shader, &renderer, &el);
                }
            }
            el.ui.draw();
        }
    }
}

#[derive(Copy, Clone)]
struct GrassVertex {
    pos: Vec3,
}

impl GrassVertex {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            pos: vec3(x, y, z),
        }
    }

    fn to_regular_vertex(&mut self) -> Vertex {
        Vertex::new(self.pos, Vec4::ONE, Vec2::ONE, Vec3::ONE)
    }

    fn to_grass_vertex(v: Vertex) -> Self {
        GrassVertex {
            pos: v.position,
        }
    }
}

#[derive(Copy, Clone)]
struct GInstanceData {
    pos: Vec3
}

struct GrassMesh {
    vertices: Vec<GrassVertex>,
    indices: Vec<u32>,

    pub VAO: u32,
    EBO: u32,
    VBO: u32,

    avg_pos: Vec3,

    pub instance_buffer: u32,

    pub instance_data: Vec<GInstanceData>,
}

impl GrassMesh {
    pub fn new(vertices: &Vec<GrassVertex>, indices: &Vec<u32>, i_data: &Vec<GInstanceData>) -> Self {
        let mut mesh = Self {
            vertices: vertices.to_vec(), indices: indices.to_vec(),
            VAO: 0, VBO: 0, EBO: 0, instance_buffer: 0,
            instance_data: i_data.to_vec(),
            avg_pos: Vec3::ZERO,
        };

        mesh
    }

    pub unsafe fn build(&mut self) {
        GenVertexArrays(1, &mut self.VAO);
        GenBuffers(1, &mut self.VBO);
        GenBuffers(1, &mut self.EBO);
        
        BindVertexArray(self.VAO);
        
        bind_buffer!(ARRAY_BUFFER, self.VBO, self.vertices);
        bind_buffer!(ELEMENT_ARRAY_BUFFER, self.EBO, self.indices);
        gen_attrib_pointers!(
            GrassVertex, 
                0 => pos: 3
        );
        
        GenBuffers(1, &mut self.instance_buffer);
        bind_buffer!(ARRAY_BUFFER, self.instance_buffer, self.instance_data);
        
        gen_attrib_pointers!(GInstanceData, 1 => pos: 3);

        VertexAttribDivisor(1, 1); 
        
        BindVertexArray(0);
    }

    pub unsafe fn draw(&self, shader: &Shader, r: &Renderer, el: &EventLoop) {
        BindVertexArray(self.VAO);

        shader.use_shader();
        r.camera.send_uniforms(&shader);
        shader.uniform_1f(cstr!("time"), el.time);
        DrawElementsInstanced(
            TRIANGLES, 
            self.indices.len() as i32, 
            UNSIGNED_INT, ptr::null(), 
            self.instance_data.len() as i32
        );
        BindVertexArray(0);
        UseProgram(0);
    }
}

// load geometry for the grass
fn grass_geometry(modelpath: &str) -> (Vec<GrassVertex>, Vec<u32>) {
    let mut model = Model::new(modelpath);
    let vertices = &model.meshes[0].vertices;
    
    let mut translated_vertices = vec![];
    
    for vertex in vertices {
        let mut v = *vertex;
        v.position /= 5.0; // make them smaller 
        translated_vertices.push(GrassVertex::to_grass_vertex(v));
    }
        
    let indices = &model.meshes[0].indices;
    (translated_vertices, indices.to_vec())
}

fn grid(y_pos: impl Fn(f32, f32) -> f32, x_segments: usize, y_segments: usize, cell_size: f32) -> tiny_game_framework::Mesh {
    let mut vertices = vec![];
    let mut indices = vec![];

    for y in 0..y_segments {
        for x in 0..x_segments {
            let fx = x as f32 * cell_size;
            let fy = y as f32 * cell_size;
            let curr_pos = vec3(fx, y_pos(fx, fy), fy);
            
            vertices.push(
                Vertex::new(curr_pos, Vec4::ONE, Vec2::ONE, Vec3::ONE),
            );

            if x < x_segments && y < y_segments {
                let base_index = (y * (x_segments + 1) + x) as u32;
                indices.push(base_index);
                indices.push(base_index + (x_segments + 1) as u32);
                indices.push(base_index + 1);

                indices.push(base_index + 1);
                indices.push(base_index + (x_segments + 1) as u32);
                indices.push(base_index + (x_segments + 2) as u32);
            }
        }
    }

    Mesh::new(&vertices, &indices)
}

fn populate_with_grass(in_vertices: &Vec<Vertex>, pos: Vec3) -> (GrassMesh, GrassMesh) {
    let (vertices_high_lod, indices_high_lod) = grass_geometry("assets/models/blade.obj");
    let (vertices_low_lod, indices_low_lod) = grass_geometry("assets/models/tri.obj");

    let mut grass_instance_data = vec![]; 
    
    let mut current_avg_pos = Vec3::ZERO;

    for vertex in in_vertices {
        let g_pos = (vertex.position / 600.0) + pos;
        grass_instance_data.push(GInstanceData {pos: g_pos});
        current_avg_pos += g_pos;
    }

    current_avg_pos = current_avg_pos / in_vertices.len() as f32;

    let mut grass_high = GrassMesh::new(&vertices_high_lod, &indices_high_lod, &grass_instance_data);
    grass_high.avg_pos = current_avg_pos;
    let mut grass_low = GrassMesh::new(&vertices_low_lod, &indices_low_lod, &grass_instance_data);
    grass_low.avg_pos = current_avg_pos;

    unsafe { grass_high.build(); grass_low.build(); };

    (grass_high, grass_low)
}
