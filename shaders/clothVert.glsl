#version 400

layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec3 vertexNormal;
layout (location = 2) in vec2 texCoord;
layout (location = 3) in int vertexFixed;

out vec3 VPosition;
out vec3 VNormal;
flat out int VvertexFixed;
out vec2 VTexCoords;

//out vec3 GPosition;
//smooth out vec3 GNormal;
//out vec2 GTexCoords;

uniform mat4 MV;
uniform mat3 normalMatrix;
uniform mat4 MVP;


void main(){
   VTexCoords = texCoord;
   VvertexFixed = vertexFixed;
   VNormal = normalize(normalMatrix * vertexNormal);
   VPosition = vec3(MV * vec4(vertexPosition,1.0));
   gl_Position = MVP * vec4(vertexPosition,1.0);
}
