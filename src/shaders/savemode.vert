#version 410

layout(location = 0) in vec3 vertex_position;
//layout(location = 1) in vec3 vertex_color;
layout(location = 2) in vec2 vertex_uv;

uniform mat4 MVP;

out vec3 fragment_color;
flat out int vertex_id;
out vec2 uv;

void main()
{                   
	gl_Position = MVP * vec4(vertex_position, 1.0);
	vertex_id = gl_VertexID ;  
	//fragment_color = vertex_color; 
	uv = vertex_uv;
}

