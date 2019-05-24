in vec4 position;

//uniform float zmin;
//uniform float zmax;

void main()
{
	//float ndcDepth = (2.0 * gl_FragCoord.z - gl_DepthRange.near - gl_DepthRange.far) / (gl_DepthRange.far - gl_DepthRange.near);
	//float clipDepth = ndcDepth / gl_FragCoord.w;
	//gl_FragColor = vec4((clipDepth * 0.5) + 0.5); 
	gl_FragColor = vec4(gl_FragCoord.z, gl_FragCoord.z, gl_FragCoord.z, 1.0);
}

