#version 400

in vec3 GNormal;
in vec3 GPosition;
in vec4 GvertexFixed;
noperspective in vec3 GEdgeDistance;
in vec2 GTexCoord;

struct lightInfo{
   vec4 position;
   vec3 intensity;
};

uniform lightInfo light;

uniform vec3 Kd;
uniform vec3 Ka;
uniform vec3 Ks;
uniform float shininess;
uniform sampler2D tex;

out vec4 fragColour;

vec3 ads(){
   vec3 n = normalize(GNormal);
   vec3 s = (normalize(light.position) - vec4(GPosition,1.0)).xyz;
   vec3 v = normalize(vec3(-GPosition));
   vec3 r = reflect(-s, n);
   vec3 h = normalize(v + s);
   return light.intensity * (Ka + Kd * max(dot(s,n),0.0)+ Ks * pow(max(dot(h, n), 0.0), shininess));
}

void  main(){
    vec4 color = vec4(ads()*texture(tex, GTexCoord).xyz,1.0);
    //if vertex is fixed colour it red

    //find smallest distance
    float d = min(GEdgeDistance.x,GEdgeDistance.y);
    d = min(d, GEdgeDistance.z);

    //determin mix factor with line
    float lineWidth = 0.001;
    float mixVal = smoothstep(lineWidth-(lineWidth/2), lineWidth+(lineWidth/2), d);

    //mix with our line colour in this case black
    color = mix(vec4(0.0,0.0,0.0,1.0), color,mixVal);
    fragColour = mix(color,GvertexFixed,GvertexFixed.a);
    //fragColour = GvertexFixed;
}
