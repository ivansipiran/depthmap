#version 450
in vec3 v_coord;
out vec4 position;
uniform mat4 m, v, p; //Matrices de transformacion, vista y proyeccion

void main(){
	mat4 mvp = p*v*m;
	position =  mvp * vec4(v_coord, 1.0f);
	gl_Position = vec4(position);
}

