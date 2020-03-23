#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertex_position;
//layout(location = 1) in vec3 vertex_color;
layout(location = 2) in vec2 vertex_uv;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;
//out vec3 fragment_color;
flat out int vertex_id;
out vec2 uv;

void main(){
	// Output position of the vertex, in clip space : MVP * position
	gl_Position =  MVP * vec4(vertex_position, 1);
	//fragment_color = vertex_color;
	vertex_id = gl_VertexID;
	uv = vertex_uv;
}

