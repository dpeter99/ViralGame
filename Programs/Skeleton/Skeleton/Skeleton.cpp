//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
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

#define TESSELATION_LEVEL 20

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
template<class T> Dnum<T> exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> sin(Dnum<T> g) { return Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> cos(Dnum<T> g) { return Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> tan(Dnum<T> g) { return sin(g) / cos(g); }
template<class T> Dnum<T> sinh(Dnum<T> g) { return Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> cosh(Dnum<T> g) { return Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> tanh(Dnum<T> g) { return sinh(g) / cosh(g); }
template<class T> Dnum<T> log(Dnum<T> g) { return Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> pow(Dnum<T> g, float n) { return Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d); }

typedef Dnum<vec2> Dnum2;

vec3 vec4_3(vec4 a) { return vec3(a.x, a.y, a.z); };

#pragma endregion

class Material;
class Light;

class Shader;
class Light;
class Scene;

class LightData;


struct RenderState
{
	mat4 MVP;
	mat4 M;
	mat4 Minv;
	mat4 V;
	mat4 P;

	Material* material;
	std::vector<LightData> lights;
	Texture* texture;
	vec3 camPos;
};



#pragma region BaseObject

/**
 * \brief This is the base class for every class in the Engine that uses runtime reflection.
 * Currently it provides a runtime TypeID and TypeName witch can be accesed as static and as class memebers.
 * The ID is a int type number witch is generated incramently, on the first call to get a type.
 * Each class that inherits from this or it's parent inheris form it must implement the
	SHObject::GetType and SHObject::GetTypeId methodes and make it's own static methodes.
	To make it easier a standard implementation of these can be used with the SHObject_Base() macro
	witch implements all of these functions. It uses the typeid().name of the class.
 */
class SHObject
{
protected:
	/**
	 * \brief Generates a new UID for each call
	 * \return the next Unique ID that was just generated
	 */
	static uint64_t GenerateId() noexcept
	{
		static uint64_t count = 0;
		return ++count;
	}

public:
	/**
	 * \brief Returns the top level class type name of the object
	 * \return The class Class name as a string
	 */
	virtual const std::string& GetType() const = 0;
	/**
	 * \brief Gets the top level type ID
	 * \return UID of the class
	 */
	virtual const uint64_t GetTypeId() const = 0;

	virtual ~SHObject() = default;
};


/**
 * \brief Macro to make the override functions of SHObject. This should be added in each derived class
 * \param type The type of the class
 */
#define SHObject_Base(type)	\
public: \
	static const std::string& Type()				{ static const std::string t = #type; return t; } \
	static uint64_t TypeId()						{ static const uint64_t id = GenerateId(); return id; } \
	const std::string& GetType() const override		{ return Type();  } \
	const uint64_t GetTypeId() const override		{ return  type::TypeId(); } \
private:

#pragma endregion


class Material
{
public:
	vec3 kd = vec3(1, 1, 1);
	vec3 ks;
	vec3 ka;

	float shiny;

	Texture* text;
	
	Shader* shader;

	void Bind(RenderState& state);
};


class Shader : public GPUProgram
{
public:
	Shader(){};
	~Shader(){};
	
	virtual void Bind(const RenderState& data) = 0;

	void setUniformMaterial(const Material& material, const std::string& name)
	{
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shiny, name + ".shininess");
	}

	void setUniformLight(const LightData& light, const std::string& name);
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
		}
	)";
#pragma endregion Shader
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(const RenderState& state) {
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
		std::vector<vec4> image(width * height);
		const vec4 yellow(1, 1, 0, 1), blue(0, 0, 1, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		}
		create(width, height, image, GL_NEAREST);
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

	virtual void Draw() = 0;
};

class ParamSurface : public Geometry
{
	struct VertexData
	{
		vec3 pos, normal;
		vec2 textcoord;
	};

	int vertexPerStrip;
	int stripCount;

public:
	
	ParamSurface() : vertexPerStrip(0), stripCount(0)
	{

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

		glBufferData(GL_ARRAY_BUFFER, vertexPerStrip * stripCount * sizeof(VertexData), &vertexData[0], GL_STATIC_DRAW);
		//Vertex atribute array enable
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, pos));
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, textcoord));

	}

	void Draw()
	{
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

		X = cos(U) * sin(V);
		Y = sin(U) * sin(V);
		Z = cos(V);
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
		V = V * 2 * M_PI;
		X = cos(V) / cosh(U);
		Y = sin(V) / cosh(U);
		Z = U - tanh(U);
	}
};

#pragma endregion Entities


#pragma region Entities

class Entity : public SHObject
{
	SHObject_Base(Entity)
protected:
	vec3 pos;
	vec3 scale;
	Quaternion rotation;
public:
	virtual void init(Scene* scene) = 0;

#pragma region Getters/Setters
	
	void setPos(vec3 p)
	{
		pos = p;
	}

	vec3 getPos()
	{
		return pos;
	}
	
#pragma endregion
	
	virtual void GetModelTransform(mat4& M, mat4& Minv)
	{
		M = ScaleMatrix(scale) * RotationMatrix(rotation.w, vec4_3(rotation)) * TranslateMatrix(pos);
		Minv = TranslateMatrix(-pos) * RotationMatrix(-rotation.w, vec4_3(rotation)) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

};


class Camera : public Entity
{
	SHObject_Base(Camera)

		vec3 wLookAt;
	vec3 wVup;
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

	~Camera()
	{

	};

	mat4 getViewMatrix()
	{
		vec3 w = normalize(pos - wLookAt);
		vec3 u = normalize(cross(wVup, w));
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
		float a = tan(fov / 2);

		return mat4(
			1 / (a * aspect_ratio), 0, 0, 0,
			0, 1 / a, 0, 0,
			0, 0, -(near_plane + far_plane) / (far_plane - near_plane), -1,
			0, 0, -2 * (near_plane * far_plane) / (far_plane - near_plane), 0
			);
	}

	void init(Scene* scene) override;
};

class Light : public Entity
{
	SHObject_Base(Light)
public:
	vec3 La, Le;
	vec4 wLightPos; // With this, it can be at any point, even at infinite distance.....

};


class Renderer : public Entity
{
	SHObject_Base(Renderer)
private:
	Geometry* shape;
	Material* mat;
public:

	Renderer(Geometry* s, Material* m) : shape(s), mat(m)
	{

	}

	void init(Scene* scene) override;

	void Render(RenderState state)
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

#pragma endregion Entities

class LightData
{
public:
	vec3 La, Le;
	vec4 wLightPos; // With this, it can be at any point, even at infinite distance.....

};

class Scene
{
	std::vector<Entity*> entities;
	Camera* mainCamera;

	std::vector<LightData> lights;
public:
	void addEntity(Entity* e)
	{
		entities.push_back(e);
		e->init(this);
		if (e->GetTypeId() == Camera::TypeId() && mainCamera == nullptr)
		{
			mainCamera = (Camera*)e;
		}
	}

	const std::vector<Entity*>& getObjects() const
	{
		return entities;
	}

	Camera* getMainCam() const
	{
		return mainCamera;
	}

	const std::vector<LightData>& getLightData()
	{
		return lights;
	}
};

class Engine
{
	Scene* activeScene;

public:
	void Init()
	{
		glViewport(0, 0, windowWidth, windowHeight);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
	}

	void LoadScene(Scene* s)
	{
		activeScene = s;
	}

	void Render()
	{
		glClearColor(0, 0, 0, 0);     // background color
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

		Camera* cam = activeScene->getMainCam();
		
		RenderState state;
		state.camPos = cam->getPos();
		state.V = cam->getViewMatrix();
		state.P = cam->getProjectonMatrix();
		state.lights = activeScene->getLightData();
		
		for (Entity* obj : activeScene->getObjects())
		{
			if(obj->GetTypeId() == Renderer::TypeId())
			{
				((Renderer*)obj)->Render(state);
			}
		}
		
		glutSwapBuffers(); // exchange buffers for double buffering
	}
};



#pragma region Late Function declaration

void Shader::setUniformLight(const LightData& light, const std::string& name)
{
	setUniform(light.La, name + ".La");
	setUniform(light.Le, name + ".Le");
	setUniform(light.wLightPos, name + ".wLightPos");
}

void Camera::init(Scene* scene)
{
	//scene->addEntity(this);
}

void Renderer::init(Scene* scene)
{
	//scene->addEntity(this);
}

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

	Scene* s = new Scene();

	Texture* tex0 = new CheckerBoardTexture(2, 2);
	
	Shader* phongShader = new PhongShader();
	
	Material* material0 = new Material;
	material0->kd = vec3(0.6f, 0.4f, 0.2f);
	material0->ks = vec3(4, 4, 4);
	material0->ka = vec3(0.1f, 0.1f, 0.1f);
	material0->shiny = 100;
	material0->shader = phongShader;
	material0->text = tex0;

	Geometry* sphere = new Sphere();

	Renderer* obj1 = new Renderer(sphere,material0);
	s->addEntity(obj1);

	Camera* cam = new Camera();
	cam->setPos(vec3(0, 10, 0));
	s->addEntity(cam);


	engine.LoadScene(s);
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
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
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
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}