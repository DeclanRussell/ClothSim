#version 400
// Taken from OpenGL 4.0 shader cookbook. (What a hell of a book that is!)
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

out vec3 GNormal;
out vec3 GPosition;
out vec2 GTexCoord;
out vec4 GvertexFixed;
noperspective out vec3 GEdgeDistance;

in vec3 VNormal[];
in vec3 VPosition[];
flat in int VvertexFixed[];
in vec2 VTexCoords[];

void main(void)
{
    //create our points
    vec3 p0 = vec3(gl_in[0].gl_Position/gl_in[0].gl_Position.w);
    vec3 p1 = vec3(gl_in[1].gl_Position/gl_in[1].gl_Position.w);
    vec3 p2 = vec3(gl_in[2].gl_Position/gl_in[2].gl_Position.w);

    //find our altitues (ha,hb,hc)
    float a = length(p1-p2);
    float b = length(p2-p0);
    float c = length(p1-p0);
    float alpha = acos( (b*b + c*c - a*a) / (2.0*b*c) );
    float beta = acos( (a*a + c*c -b*b) / (2.0*a*c) );
    float ha = abs(c * sin(beta) );
    float hb = abs(c * sin(alpha));
    float hc = abs(b * sin(alpha));

    //send all our information to our vertex shader
    GEdgeDistance = vec3(ha,0,0);
    GNormal = VNormal[0];
    GPosition = VPosition[0];
    if(VvertexFixed[0]>0){
        GvertexFixed = vec4(1,0,0,1);
    }
    else
    {
        GvertexFixed = vec4(0,0,0,0);
    }
    GTexCoord = VTexCoords[0];
    gl_Position = gl_in[0].gl_Position;
    EmitVertex();

    GEdgeDistance = vec3(0,hb,0);
    GNormal = VNormal[1];
    GPosition = VPosition[1];
    if(VvertexFixed[1]>0){
        GvertexFixed = vec4(1,0,0,1);
    }
    else
    {
        GvertexFixed = vec4(0,0,0,0);
    }
    GTexCoord = VTexCoords[1];
    gl_Position = gl_in[1].gl_Position;
    EmitVertex();

    GEdgeDistance = vec3(0,0,hc);
    GNormal = VNormal[2];
    GPosition = VPosition[2];
    if(VvertexFixed[2]>0){
        GvertexFixed = vec4(1,0,0,1);
    }
    else
    {
        GvertexFixed = vec4(0,0,0,0);
    }
    GTexCoord = VTexCoords[2];
    gl_Position = gl_in[2].gl_Position;
    EmitVertex();

    EndPrimitive();
}
