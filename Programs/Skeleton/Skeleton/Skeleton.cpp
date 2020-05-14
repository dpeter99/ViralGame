//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

typedef unsigned int GL_OBJECT;

typedef vec4 Quaternion;

#define TESSELATION_LEVEL 40

#pragma region Math

template<class T> struct Dnum
{
	float f;
	T d;
	Dnum(float f0 = 0, T d0 = T(0)) : f(f0), d(d0) {}

	Dnum operator+(Dnum other) { return Dnum(f + other.f, d + other.d); }
	Dnum operator-(Dnum other) { return Dnum(f - other.f, d - other.d); }

	Dnum operator*(Dnum other) { return Dnum(f * other.f, f * other.d + d * other.f); }
	Dnum operator/(Dnum other) { return Dnum(f / other.f, (other.f * d - other.d * f) / other.f / other.f); }
};

//Basic functions for derivation
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T> g) { return Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T> g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) { return Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d); }

typedef Dnum<vec2> Dnum2;

vec3 vec4_3(vec4 a) { return vec3(a.x, a.y, a.z); };

bool operator==(const vec3& a, const vec3& b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

bool operator!=(const vec3& a, const vec3& b)
{
	return !(a == b);
}


bool operator==(const vec4& a, const vec4& b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

bool operator!=(const vec4& a, const vec4& b)
{
	return !(a == b);
}


inline vec3 normalize_fix(const vec3& v) { return (v != vec3(0, 0, 0)) ? v / length(v) : vec3(0, 0, 0); }

inline float length_squared(const vec3& v) { return v.x * v.x + v.y * v.y + v.z * v.z; }


inline float length_squared(const vec4& v1) { return (v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w); }

inline float length(const vec4& v1) { return sqrtf(length_squared(v1)); }

inline vec4 normalize(const vec4& value)
{
	Quaternion ans;

	float ls = value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w;

	float invNorm = 1.0f / sqrtf(ls);

	ans.x = value.x * invNorm;
	ans.y = value.y * invNorm;
	ans.z = value.z * invNorm;
	ans.w = value.w * invNorm;

	return ans;
}

inline Quaternion FromTo(vec3 from, vec3 to)
{
	const float NormAB = sqrtf(length_squared(from) * length_squared(to));

	Quaternion r;
	vec3 a = cross(from, to);
	r.x = a.x;
	r.y = a.y;
	r.z = a.z;

	r.w = dot(from, to) + NormAB;

	return normalize(r);
}

#pragma endregion

class Material;
class Light;

class Shader;
class Light;
class Scene;


struct RenderState
{
	mat4 MVP;
	mat4 M;
	mat4 Minv;
	mat4 V;
	mat4 P;

	Material* material = nullptr;
	std::vector<Light> lights;
	Texture* texture = nullptr;
	vec3 camPos;
};

class Light
{
public:
	vec3 La, Le;
	vec4 wLightPos; // With this, it can be at any point, even at infinite distance.....
};

class Material
{
public:
	vec3 kd = vec3(1, 1, 1);
	vec3 ks;
	vec3 ka;

	float shiny = 0;

	Texture* text = nullptr;

	Shader* shader = nullptr;

	void Bind(RenderState& state);
};



class Shader : public GPUProgram
{
public:
	Shader() {};
	~Shader() {};

	virtual void Bind(const RenderState& data) = 0;

	void setUniformMaterial(const Material& material, const std::string& name)
	{
		//printf("Binding Material data\n");
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shiny, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name)
	{
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

class PhongShader : public Shader
{
#pragma region Shader
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + 
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
			//fragmentColor = vec4(N,1);
		}
	)";
#pragma endregion Shader
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(const RenderState& state) {
		//printf("Binding Phong shader\n");

		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.camPos, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};



class CheckerBoardTexture : public Texture
{
public:
	CheckerBoardTexture(const int width = 0, const int height = 0) : Texture() {
		//printf("Created Checker Board Texture\n");

		std::vector<vec4> image(width * height);
		const vec4 yellow(1, 1, 0, 1), blue(0, 0, 1, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		}
		create(width, height, image, GL_NEAREST);
	}
};

class StripesTexture : public Texture
{
public:
	StripesTexture(const int Twidth = 0, const int Theight = 0, const int Sheight = 1) : Texture() {
		//printf("Created Checker Board Texture\n");

		std::vector<vec4> image(Twidth * Theight);
		const vec4 yellow(1, 1, 0, 1), blue(0, 0, 1, 1);
		for (int x = 0; x < Twidth; x++)
			for (int y = 0; y < Theight; y++) {
				image[y * Twidth + x] = (y / Sheight) % 2 == 0 ? yellow : blue;
			}
		create(Twidth, Theight, image, GL_NEAREST);
	}
};

#pragma region Geometry

class Geometry
{
protected:
	GL_OBJECT vertex_array;
	GL_OBJECT vertex_buffer;
public:
	Geometry()
	{
		glGenVertexArrays(1, &vertex_array);
		glBindVertexArray(vertex_array);
		glGenBuffers(1, &vertex_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
	}

	~Geometry()
	{
		glDeleteBuffers(1, &vertex_buffer);
		glDeleteVertexArrays(1, &vertex_array);
	}

	virtual void Tick(float deltaTime) {};

	virtual void Draw() = 0;

	virtual vec3 GetNormalAt(vec2 pos) = 0;

	virtual vec3 GetPosAt(vec2 pos) = 0;
};

class ParamSurface : public Geometry
{
	struct VertexData
	{
		vec3 pos, normal;
		vec2 textcoord;
	};

	int vertexPerStrip;
	unsigned int stripCount;

public:

	ParamSurface() : vertexPerStrip(0), stripCount(0)
	{

	}

	virtual vec3 GetNormalAt(vec2 pos)
	{
		return GetVertexData(pos.x, pos.y).normal;
	}

	virtual vec3 GetPosAt(vec2 pos)
	{
		vec3 a = GetVertexData(pos.x, pos.y).pos;
		return a;
	}


	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GetVertexData(float u, float v)
	{
		VertexData res;
		res.textcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0));
		Dnum2 V(v, vec2(0, 1));

		eval(U, V, X, Y, Z);

		res.pos = vec3(X.f, Y.f, Z.f);

		vec3 drdU(X.d.x, Y.d.x, Z.d.x);
		vec3 drdV(X.d.y, Y.d.y, Z.d.y);

		res.normal = cross(drdU, drdV);

		return res;
	}

	void create(int N = TESSELATION_LEVEL, int M = TESSELATION_LEVEL)
	{
		//printf("Creating the Geometry (Param)\n");

		vertexPerStrip = (M + 1) * 2;
		stripCount = N;
		std::vector<VertexData> vertexData;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j <= M; j++)
			{
				vertexData.push_back(GetVertexData((float)j / M, (float)i / N));
				vertexData.push_back(GetVertexData((float)j / M, (float)(i + 1) / N));
			}
		}

		glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
		glBindVertexArray(vertex_array);
		glBufferData(GL_ARRAY_BUFFER, vertexPerStrip * stripCount * sizeof(VertexData), &vertexData[0], GL_STATIC_DRAW);
		//Vertex atribute array enable
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, pos));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, textcoord));

	}

	void Draw()
	{
		//printf("Draw the Geometry (Param)\n");

		glBindVertexArray(vertex_array);
		for (unsigned int i = 0; i < stripCount; i++)
		{
			glDrawArrays(GL_TRIANGLE_STRIP, i * vertexPerStrip, vertexPerStrip);
		}
	}
};


class Sphere : public ParamSurface
{
public:
	Sphere()
	{
		create();
	}

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z)
	{
		U = U * 2.0f * (float)M_PI;
		V = V * (float)M_PI;

		X = Cos(U) * Sin(V);
		Y = Sin(U) * Sin(V);
		Z = Cos(V);
	}
};

class Tractricoid : public ParamSurface
{
public:
	Tractricoid()
	{
		create();
	}

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z)
	{
		const float height = 3.0f;
		U = U * height;
		V = V * (2.0f * (float)M_PI);
		X = Cos(V) / Cosh(U);
		Y = Sin(V) / Cosh(U);
		Z = U - Tanh(U);
	}
};

class AnimatedSurface : public ParamSurface
{
protected:
	float t = 0;
	float dt = 0;

public:
	void Tick(float deltaTime)
	{
		dt = deltaTime;
		t += deltaTime;
		create();
	}
};

class WavySphere : public AnimatedSurface
{
public:
	WavySphere()
	{
		create();
	}

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z)
	{
		const float anim = t / (10.0f * (float)M_PI);

		U = U * 2.0f * (float)M_PI;
		V = V * (float)M_PI;

		float slow_anim = anim / 10;
		float magnitude = (1.0f / 20) * (abs(sinf(slow_anim)));
		Dnum2 height = Sin(V * 15 + anim) * magnitude;
		height = height + 1;
		//X = (cos(U) * sin(V)) + (sin(V*5 + anim)/4);
		//Y = (sin(U) * sin(V)) + (sin(V*5 + anim)/4);
		//height = 1;

		X = (Cos(U) * Sin(V)) * height;// ((sin(V * 10 + anim / 10) + 1) / 5));
		Y = (Sin(U) * Sin(V)) * height;

		Z = Cos(V);// *((sin(V * 10 + anim / 10) / 10) + 1);
	}
};

class TriangleRecurive : public Geometry
{
	unsigned int vertexCount;
	float time = 0;

	int iterMax = 2;

	struct VertexData
	{
		vec3 pos, normal;
		vec2 textcoord;

		VertexData(vec3 p, vec3 norm) : pos(p), normal(norm)
		{

		}
	};

	struct Triangle
	{
		vec3 a;
		vec3 b;
		vec3 c;

		vec3 norm;

		Triangle() = default;

		Triangle(const vec3& a, const vec3& b, const vec3& c, const vec3& n)
			: a(a),
			b(b),
			c(c),
			norm(n)
		{
		}
	};

public:
	TriangleRecurive()
	{
		create();
	};

	void Tick(float deltaTime) override
	{
		time += deltaTime;
		create();
	}
	
	/*
	std::vector<VertexData> maketeface(vec3 A, vec3 B, vec3 C, vec3 center, int iter)
	{
		std::vector<VertexData> res;

		//calc the normal of the face:
		vec3 norm_long = ((A + B + C) / 3) - center;
		vec3 norm = normalize(norm_long);



		vec3 AB = (A + B) / 2;
		vec3 BC = (B + C) / 2;
		vec3 AC = (A + C) / 2;

		VertexData A_data(A, norm);
		VertexData B_data(B, norm);
		VertexData C_data(C, norm);
		VertexData AB_data(AB, norm);
		VertexData AC_data(AC, norm);
		VertexData BC_data(BC, norm);

		//Make the center
		float x = time / 100;
		float height = sinf(x) + abs(sinf(x)) + 2;
		height *= length(norm_long);

		vec3 end = ((A + B + C) / 3) + (norm * height);

		if (iter <= iterMax - 1) {

			iter++;

			std::vector<VertexData> sub = maketeTetraeder(A, AC, AB, end, iter);
			res.insert(res.end(), sub.begin(), sub.end());

			sub = maketeTetraeder(AC, BC, C, end, iter);
			res.insert(res.end(), sub.begin(), sub.end());

			sub = maketeTetraeder(AB, B, BC, end, iter);
			res.insert(res.end(), sub.begin(), sub.end());

		}
		else
		{
			//Triangles:
			// A - AC - AB
			res.push_back(A_data);
			res.push_back(AC_data);
			res.push_back(AB_data);
			// AC - BC - C
			res.push_back(AC_data);
			res.push_back(BC_data);
			res.push_back(C_data);
			// AB - B -BC
			res.push_back(AB_data);
			res.push_back(B_data);
			res.push_back(BC_data);
		}



		if (iter <= iterMax) {
			iter++;
			std::vector<VertexData> sub = maketeTetraeder(AB, AC, BC, end, iter);
			res.insert(res.end(), sub.begin(), sub.end());
		}
		else
		{
			res.push_back(AB_data);
			res.push_back(BC_data);
			res.push_back(AC_data);
		}

		return res;
	}

	std::vector<VertexData> maketeTetraeder(vec3 A, vec3 B, vec3 C, vec3 D, int iter)
	{
		vec3 center = (A + B + C + D) / 4;

		std::vector<VertexData> model;
		std::vector<VertexData> side;
		if (iter == 0) {
			side = maketeface(A, B, C, center, iter);
			model.insert(model.end(), side.begin(), side.end());
		}

		side = maketeface(A, B, D, center, iter);
		model.insert(model.end(), side.begin(), side.end());

		side = maketeface(A, C, D, center, iter);
		model.insert(model.end(), side.begin(), side.end());

		side = maketeface(B, C, D, center, iter);
		model.insert(model.end(), side.begin(), side.end());

		return model;
	}
	*/
	
	std::vector<VertexData> make()
	{
		vec3 A(1, -(1.73f / 2), 0);
		vec3 B(-1, -(1.73f / 2), 0);
		vec3 C(0, 1.73 / 2, 0);
		vec3 D(0, 0, 1.73);

		vec3 center = (A + B + C + D) / 4;



		//add the base 3
		std::vector <Triangle> triangles;

		vec3 norm_ABC = normalize(((A + B + C) / 3) - center);
		triangles.push_back(Triangle(A, B, C, norm_ABC));

		vec3 norm_ABD = normalize(((A + B + D) / 3) - center);
		triangles.push_back(Triangle(A, B, D, norm_ABD));

		vec3 norm_ACD = normalize(((A + C + D) / 3) - center);
		triangles.push_back(Triangle(A, C, D, norm_ACD));

		vec3 norm_BCD = normalize(((B + C + D) / 3) - center);
		triangles.push_back(Triangle(B, C, D, norm_BCD));


		for (int i = 0; i < iterMax; i++)
		{
			std::vector <Triangle> new_list;
			for (Triangle t : triangles)
			{
				//        B
				//       / \
				//      /   \
				// AB  /_____\  BC
				//    / \    /\
				//   /   \  /  \
				//	/     \/    \
				// A-------------C
				//	      AC

				vec3 AB = (t.a + t.b) / 2;
				vec3 BC = (t.b + t.c) / 2;
				vec3 AC = (t.a + t.c) / 2;

				//Triangles:
				// A - AC - AB
				new_list.push_back(Triangle(t.a, AC, AB, t.norm));
				// AC - BC - C
				new_list.push_back(Triangle(AC, BC, t.c, t.norm));
				// AB - B -BC
				new_list.push_back(Triangle(AB, t.b, BC, t.norm));


				float x = time / 100;
				//float height = sinf(x) + abs(sinf(x)) + 2;
				//height *= length(t.norm);

				float height = sqrtf(3) / 2 * length(AB - AC);
				height *= (sinf(x) + abs(sinf(x)) / 2) + 1;

				//printf("%f \n", length((normalize(t.norm) * height)));
				
				//new top
				vec3 end = ((AB + AC + BC) / 3) + (normalize(t.norm) * height);


				new_list.push_back(createSubFace(AB, BC, end, AC));
				new_list.push_back(createSubFace(AB, AC, end, BC));
				new_list.push_back(createSubFace(AC, BC, end, AB));
			}
			triangles = new_list;
		}

		std::vector <VertexData> verts;
		for (Triangle t : triangles)
		{
			verts.push_back(VertexData(t.a, t.norm));
			verts.push_back(VertexData(t.b, t.norm));
			verts.push_back(VertexData(t.c, t.norm));
		}
		return verts;
	}

	Triangle createSubFace(vec3 A, vec3 B, vec3 C, vec3 D)
	{
		//Calculate the face normal
		vec3 AB = A - B;
		vec3 AC = A - C;
		vec3 back_dir = D - A;

		vec3 norm =  cross(AB, AC);
		if (dot(norm, back_dir) > 0)
		{
			norm = norm * -1;
		}

		return Triangle(A, B, C, normalize(norm));
	}

	void create(int level = 2)
	{
		//std::vector<VertexData> vertexData = maketeTetraeder(vec3(1, -(1.73f / 2), 0), vec3(-1, -(1.73f / 2), 0), vec3(0, 1.73 / 2, 0), vec3(0, 0, 1.73), 0);
		std::vector<VertexData> vertexData = make();
		vertexCount = vertexData.size();

		glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
		glBindVertexArray(vertex_array);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(VertexData), &vertexData[0], GL_STATIC_DRAW);
		//Vertex atribute array enable
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, pos));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, textcoord));

	}

	void Draw()
	{
		//printf("Draw the Geometry (Param)\n");

		glBindVertexArray(vertex_array);


		glDrawArrays(GL_TRIANGLES, 0, vertexCount);

	}

public:
	vec3 GetNormalAt(vec2 pos) override { return vec3(0, 0, 0); };
	vec3 GetPosAt(vec2 pos) override { return vec3(0, 0, 0); }
	;
};

#pragma endregion Geometry


#pragma region Entities

class Entity
{
protected:
	vec3 pos = vec3(0, 0, 0);
	vec3 scale = vec3(1, 1, 1);
	Quaternion rotation = vec4(0, 1, 0, 0);

	Entity* parent = nullptr;

	bool manualModelMatrix = false;
	mat4 M;
	mat4 Minv;

	void recalcMatrixes()
	{
		if (manualModelMatrix == false) {
			M = ScaleMatrix(scale) * RotationMatrix(rotation.w, vec4_3(rotation)) * TranslateMatrix(pos);
			Minv = TranslateMatrix(-pos) * RotationMatrix(-rotation.w, vec4_3(rotation)) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		}
	}

public:

	Entity()
	{
		recalcMatrixes();
	}

	virtual void tick(float time) {};

	virtual void render(RenderState state) {};

#pragma region Getters/Setters

	void setPos(vec3 p)
	{
		pos = p;
		recalcMatrixes();
	}

	vec3 getPos()
	{
		return pos;
	}

	void setScale(vec3 s)
	{
		this->scale = s;
		recalcMatrixes();
	}

	void setRot(Quaternion r)
	{
		rotation = r;
		recalcMatrixes();
	}

	void setParent(Entity* e)
	{
		parent = e;
	}

	void setMatrix(mat4 M, mat4 Minv)
	{
		this->M = M; this->Minv = Minv;
	}
#pragma endregion

	virtual void GetModelTransform(mat4& M_out, mat4& Minv_out)
	{
		M_out = M;
		Minv_out = Minv;

		if (parent != nullptr)
		{
			mat4 Mparent;
			mat4 MparentInv;
			parent->GetModelTransform(Mparent, MparentInv);
			M_out = M_out * Mparent;
			Minv_out = Minv_out * MparentInv;
		}
	}

};


class Camera : public Entity
{
private:
	vec3 wLookAt = vec3(0, 0, 0);
	vec3 wVup = vec3(0, 1, 0);
	float fov;
	float aspect_ratio;
	float near_plane;
	float far_plane;

public:
	Camera()
	{
		aspect_ratio = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		near_plane = 1;
		far_plane = 20;
	};

	mat4 getViewMatrix()
	{
		vec3 w = normalize_fix(pos - wLookAt);
		vec3 u = normalize_fix(cross(wVup, w));
		vec3 v = cross(w, u);

		return TranslateMatrix(pos * (-1))
			* mat4(
				u.x, v.x, w.x, 0,
				u.y, v.y, w.y, 0,
				u.z, v.z, w.z, 0,
				0, 0, 0, 1);
	}

	mat4 getProjectonMatrix()
	{
		float a = tanf(fov / 2);

		return mat4(
			1 / (a * aspect_ratio), 0, 0, 0,
			0, 1 / a, 0, 0,
			0, 0, -(near_plane + far_plane) / (far_plane - near_plane), -1,
			0, 0, -2 * (near_plane * far_plane) / (far_plane - near_plane), 0
			);
	}

};

class Renderer : public Entity
{
protected:
	Geometry* shape;
	Material* mat;
public:

	Renderer(Geometry* s, Material* m) :Entity(), shape(s), mat(m)
	{
		this->rotation = vec4(0, 1, 0, 0);
	}

	Geometry* getMesh() { return shape; }


	void tick(float time) override
	{
		shape->Tick(time);
	}

	void render(RenderState state) override
	{
		mat4 M, Minv;
		GetModelTransform(M, Minv);

		state.M = M;
		state.Minv = Minv;
		state.MVP = M * state.V * state.P;
		state.material = mat;
		//state.texture = tex
		mat->Bind(state);
		shape->Draw();
	}
};

//This is the virus
//It holds the base model
//And updates the tentacles each frame to align to the surface
class Virus : public Entity
{
	Material* material0;
	Material* tentacle_mat;

	Renderer* body;

	Geometry* tentacleMesh;
	std::vector<Renderer*> tentecles;

public:

	Virus() :Entity()
	{
		Texture* tex0 = new StripesTexture(1, 50, 2);

		Shader* phongShader = new PhongShader();

		material0 = new Material;
		material0->kd = vec3(0.6f, 0.4f, 0.2f);
		material0->ks = vec3(4, 4, 4);
		material0->ka = vec3(0.5f, 0.5f, 0.5f);
		material0->shiny = 100000;
		material0->shader = phongShader;
		material0->text = tex0;

		tentacle_mat = new Material;
		tentacle_mat->kd = vec3(0.6f, 0.4f, 0.2f);
		tentacle_mat->ks = vec3(4, 4, 4);
		tentacle_mat->ka = vec3(0.5f, 0.5f, 0.5f);
		tentacle_mat->shiny = 100000;
		tentacle_mat->shader = phongShader;
		tentacle_mat->text = tex0;

		Geometry* sphere = new WavySphere();
		body = new Renderer(sphere, material0);
		body->setParent(this);

		tentacleMesh = new Tractricoid();

		//tentecles.push_back(createTentacle());
		createTentacle();
	}

	void createTentacle()
	{

		//ten->setScale(vec3(0.2f, 0.2f, 0.2f));
		float N = 6; //rows Pi
		float M = 10; //	2* Pi

		for (size_t i = 0; i <= N; i++)
		{
			float v = i / N;
			float count = M * sin(v * M_PI);
			for (size_t j = 0; j <= count; j++)
			{
				float u = j / floor(count);

				Renderer* ten = new Renderer(tentacleMesh, material0);
				ten->setParent(this);

				setTentaclePos(ten, u, v);

				tentecles.push_back(ten);
			}
		}

		//setTentaclePos(ten);

		//return ten;
	}

	void setTentaclePos(Renderer* ten, float u, float v)
	{
		vec3 pos = body->getMesh()->GetPosAt(vec2(u, v));
		vec3 normal = normalize(body->getMesh()->GetNormalAt(vec2(u, v)));
		printf("Raw normal length: %f \n", length(normal));

		float scale = 0.2f;

		pos = pos;// +normal;// *(2 * scale);

		vec3 base_axis = vec3(0, 0, -1);
		vec3 axis = normalize(cross(base_axis, pos));
		float angle = dot(base_axis, normal);
		angle /= length(base_axis) * length(normal);
		angle += M_PI_2;

		//pos = pos;// +(normalize(normal));
		//ten->setPos(pos);


		//Quaternion rot = FromTo(vec3(0, 0, -1), normal);
		//rot = rot == vec4(0, 0, 0, 0) ? vec4(0, 1, 0, 0) : rot;
		//printf("Quat length: %f \n", length(rot));
		//ten->setRot(rot);

		mat4 transform = ScaleMatrix(vec3(scale, scale, scale)) * TranslateMatrix(vec3(0, 0, -(2.0f * scale))) * RotationMatrix(angle, axis) * TranslateMatrix(pos);
		mat4 transform_inv = TranslateMatrix(-pos) * RotationMatrix(-angle, axis) * TranslateMatrix(-vec3(0, 0, -(2.0f * scale))) * ScaleMatrix(vec3(1 / scale, 1 / scale, 1 / scale));

		//mat4 transform = ScaleMatrix(vec3(scale, scale, scale)) * TranslateMatrix(vec3(0, 0, -(2.0f * scale))) * RotationMatrix(rot.w, vec4_3(rot)) * TranslateMatrix(pos);
		//mat4 transform_inv = TranslateMatrix(-pos) * RotationMatrix(-rot.w, vec4_3(rot)) * TranslateMatrix(-vec3(0, 0, -(2.0f * scale))) * ScaleMatrix(vec3(1 / scale, 1 / scale, 1 / scale));

		ten->setMatrix(transform, transform_inv);
	}

	void tick(float dt)
	{
		dt /= 10;
		float r = this->rotation.w;
		r += dt * (2.0f / 180);
		this->setRot(vec4(0, 1, 0, r));

		body->tick(dt);

		for (Renderer* r : tentecles)
		{
			//setTentaclePos(r,);
		}
	}

	void render(RenderState state) override
	{
		body->render(state);

		for (auto* var : tentecles)
		{
			var->render(state);
		}
	}

};

class Antibody : public Entity
{
	Material* material0;

	Renderer* body;

public:

	Antibody()
	{
		Texture* tex0 = new StripesTexture(1, 12, 2);

		Shader* phongShader = new PhongShader();

		material0 = new Material;
		material0->kd = vec3(0.6f, 0.4f, 0.2f);
		material0->ks = vec3(4, 4, 4);
		material0->ka = vec3(0.5f, 0.5f, 0.5f);
		material0->shiny = 100000;
		material0->shader = phongShader;
		material0->text = tex0;

		Geometry* sphere = new TriangleRecurive();
		body = new Renderer(sphere, material0);
		body->setParent(this);
	}

	void tick(float dt)
	{
		dt /= 10;
		float r = this->rotation.w;
		r += dt * (2.0f / 180);
		this->setRot(vec4(0, 1, 0, r));

		body->tick(dt);
	}

	void render(RenderState state) override
	{
		body->render(state);
	}

};

#pragma endregion Entities


class Engine
{
	std::vector<Entity*> entities;
	Camera* mainCamera = nullptr;

	std::vector<Light> lights;

public:


	void Init()
	{
		glViewport(0, 0, windowWidth, windowHeight);
		glEnable(GL_DEPTH_TEST);
		//glEnable(GL_CULL_FACE);
	}

	void addEntity(Entity* e)
	{
		entities.push_back(e);
	}

	void addLight(Light e)
	{
		lights.push_back(e);
	}

	void setCamera(Camera* cam)
	{
		mainCamera = cam;
	}

	void Render()
	{
		glClearColor(0, 0, 0, 0);     // background color
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

		RenderState state;
		state.camPos = mainCamera->getPos();
		state.V = mainCamera->getViewMatrix();
		state.P = mainCamera->getProjectonMatrix();
		state.lights = lights;

		for (Entity* obj : entities)
		{
			obj->render(state);
		}

		glutSwapBuffers(); // exchange buffers for double buffering
	}

	void tick(float time)
	{
		for (Entity* var : entities)
		{
			var->tick(time);
		}
	};
};

#pragma region Late Function declaration

void Material::Bind(RenderState& state)
{
	state.material = this;
	state.texture = text;
	this->shader->Bind(state);
}

#pragma endregion

Engine engine;

// Initialization, create an OpenGL context
void onInitialization() {

	engine.Init();

	Entity* obj1 = new Antibody();
	obj1->setPos(vec3(0, 0, 0));
	engine.addEntity(obj1);


	Light light0;
	light0.wLightPos = vec4(3, 3, 3, 0);
	light0.La = vec3(0.7f, 0.7f, 0.7f);
	light0.Le = vec3(3, 3, 3);
	engine.addLight(light0);

	Camera* cam = new Camera();
	cam->setPos(vec3(0.1f, 0.1f, 5));
	engine.setCamera(cam);

}

// Window has become invalid: Redraw
void onDisplay() {
	engine.Render();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	//printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	/*
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char* buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
	*/
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	static long lastTime = 0;

	const float deltaTime = time - lastTime;
	lastTime = time;
	engine.tick(deltaTime);

	glutPostRedisplay();
}
