#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <set>
#include <string>
#include <cmath>
#include <cassert>
#include <time.h>
#include <iomanip>
#include <sstream>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "shader_utils.h"

#define M_PI 3.14159264

using namespace std;

////////////////////////////////////////////////////////////
// TYPE DEFINITIONS
////////////////////////////////////////////////////////////
float angle = 0.0;

typedef struct Vertex {
    float x, y, z;
    unsigned int index;
    float normal[3];
    int num_faces;
} Vertex;

typedef struct Face {
    Face(void) : nverts(0), verts(0) {};
    int nverts;
    Vertex **verts;
    float normal[3];
} Face;

typedef struct Mesh {
    Mesh(void) : nverts(0), verts(0), nfaces(0), faces(0) {};
    int nverts;
    Vertex *verts;
    int nfaces;
    Face *faces;
    float center[3];
    float scale[3];
} Mesh;

////////////////////////////////////////////////////////////
// GLOBAL VARIABLES
////////////////////////////////////////////////////////////

Mesh* mymesh;

int screen_width = 800, screen_height = 800;

int nLoops = 10;

GLfloat* vertex_buffer = NULL;
GLfloat* normal_buffer = NULL;
GLuint* index_buffer = NULL;
GLuint vbo_vertices;
GLuint vbo_normals;
GLuint ibo_triangles;
GLuint program;
GLint attribute_v_coord;
GLint attribute_v_normal;
GLint uniform_m, uniform_v, uniform_p;
GLint uniform_m_3x3_inv_transp;
GLint uniform_v_inv;

const int window_height = 10, window_width = 10;
const int num_seeds = screen_width * screen_height / 128;
const int descriptor_size = 32;
const int bit_diff_threshold = 2;
const int pair_count_threshold = num_seeds * 0.9;

int planeCount = 0;

////////////////////////////////////////////////////////////
// SYMMETRY DETECTION
////////////////////////////////////////////////////////////
float getValueFromBitArray(GLubyte* bits, int w, int h, int x, int y){
    return (bits[(h-y-1)*3*w + x*3+0] + bits[(h-y-1)*3*w + x*3+1] + bits[(h-y-1)*3*w + x*3+2])/3;
}

////////////////////////////////////////////////////////////
// OTHER METHODS
////////////////////////////////////////////////////////////

Mesh* readOffFile(const char *filename){
    int i;
    float bbox[2][3];

    bbox[0][0] = bbox[0][1] = bbox[0][2] = 1.0E30F;
    bbox[1][0] = bbox[1][1] = bbox[1][2] = -1.0E30F;

    // Open file
    FILE *fp;
    if (!(fp = fopen(filename, "r"))) {
        fprintf(stderr, "No se puede abrir archivo %s\n", filename);
        return 0;
    }

    // Allocate mesh structure
    Mesh *mesh = new Mesh();
    mesh->center[0] = mesh->center[1] = mesh->center[2] = 0.0;

    if (!mesh){
        fprintf(stderr, "No se puede crear memoria para archivo %s\n", filename);
        fclose(fp);
        return 0;
    }

    // Read file
    int nverts = 0;
    int nfaces = 0;
    int nedges = 0;
    int line_count = 0;
    char buffer[1024];
    while (fgets(buffer, 1023, fp)) {
        // Increment line counter
        line_count++;

        // Skip white space
        char *bufferp = buffer;
        while (isspace(*bufferp)) bufferp++;

        // Skip blank lines and comments
        if (*bufferp == '#') continue;
        if (*bufferp == '\0') continue;

        // Check section
        if (nverts == 0) {
            // Read header
            if (!strstr(bufferp, "OFF")) {
                // Read mesh counts
                if ((sscanf(bufferp, "%d%d%d", &nverts, &nfaces, &nedges) != 3) || (nverts == 0)) {
                  fprintf(stderr, "Error de sintaxis en la cabecera en la linea %d %s\n", line_count, filename);
                  fclose(fp);
                  return NULL;
                }

                // Allocate memory for mesh
                mesh->verts = new Vertex [nverts];
                assert(mesh->verts);
                mesh->faces = new Face [nfaces];
                assert(mesh->faces);
            }
        }
        else if (mesh->nverts < nverts) {
            // Read vertex coordinates
            Vertex& vert = mesh->verts[mesh->nverts++];
            vert.index = mesh->nverts - 1;
            vert.normal[0] = vert.normal[1] = vert.normal[2] = 0.0;
            vert.num_faces = 0;
            if (sscanf(bufferp, "%f%f%f", &(vert.x), &(vert.y), &(vert.z)) != 3) {
                fprintf(stderr, "Error de sintaxis en vertice en linea %d %s\n", line_count, filename);
                fclose(fp);
                return NULL;
            }

            if(vert.x < bbox[0][0]) bbox[0][0] = vert.x;
            else if(vert.x > bbox[1][0]) bbox[1][0] = vert.x;

            if(vert.y < bbox[0][1]) bbox[0][1] = vert.y;
            else if(vert.y > bbox[1][1]) bbox[1][1] = vert.y;

            if(vert.z < bbox[0][2]) bbox[0][2] = vert.z;
            else if(vert.z > bbox[1][2]) bbox[1][2] = vert.z;

        }
        else if (mesh->nfaces < nfaces) {
            // Get next face
            Face& face = mesh->faces[mesh->nfaces++];

            // Read number of vertices in face
            bufferp = strtok(bufferp, " \t");
            if (bufferp) face.nverts = atoi(bufferp);
            else {
                fprintf(stderr, "1. Error de sintaxis en cara en linea %d %s\n", line_count, filename);
                fclose(fp);
                return NULL;
            }

            //face.nverts = 3;
            // Allocate memory for face vertices
            face.verts = new Vertex *[face.nverts];
            assert(face.verts);

            bufferp = strtok(NULL, " \t");
            if(bufferp) face.verts[0] = &(mesh->verts[atoi(bufferp)]);
            else{
                fprintf(stderr, "2. Error de sintaxis en cara en linea %d  %s\n", line_count, filename);
                fclose(fp);
                return NULL;
            }

            // Read vertex indices for face
            for (i = 1; i < face.nverts; i++) {
                bufferp = strtok(NULL, " \t");
                if (bufferp) face.verts[i] = &(mesh->verts[atoi(bufferp)]);
                else {
                    fprintf(stderr, "3. Error de sintaxis en cara en linea %d %s\n", line_count, filename);
                    fclose(fp);
                    return NULL;
                }
            }

            // Compute normal for face
            face.normal[0] = face.normal[1] = face.normal[2] = 0;
            Vertex *v1 = face.verts[face.nverts-1];
            for (i = 0; i < face.nverts; i++) {
                Vertex *v2 = face.verts[i];
                face.normal[0] += (v1->y - v2->y) * (v1->z + v2->z);
                face.normal[1] += (v1->z - v2->z) * (v1->x + v2->x);
                face.normal[2] += (v1->x - v2->x) * (v1->y + v2->y);
                v1 = v2;
            }

            // Normalize normal for face
            float squared_normal_length = 0.0;
            squared_normal_length += face.normal[0]*face.normal[0];
            squared_normal_length += face.normal[1]*face.normal[1];
            squared_normal_length += face.normal[2]*face.normal[2];
            float normal_length = sqrt(squared_normal_length);
            if (normal_length > 1.0E-6) {
                face.normal[0] /= normal_length;
                face.normal[1] /= normal_length;
                face.normal[2] /= normal_length;
            }

            //Sum the face normal in the adjacent vertices
            for(i = 0; i < face.nverts; i++){
                Vertex *aux = face.verts[i];
                aux->normal[0] += face.normal[0];   aux->normal[1] += face.normal[1]; aux->normal[2] += face.normal[2];
                aux->num_faces++;
            }
        }
        else {
            // Should never get here
            fprintf(stderr, "Hay texto de mas en linea %d %s\n", line_count, filename);
            break;
        }
    }

    //Average vertex normals
    for(int i = 0; i < mesh->nverts; i++){
        Vertex* aux = &mesh->verts[i];
        aux->normal[0] /= aux->num_faces;
        aux->normal[1] /= aux->num_faces;
        aux->normal[2] /= aux->num_faces;
        float mag = sqrt(aux->normal[0] * aux->normal[0] + aux->normal[1]*aux->normal[1] + aux->normal[2]*aux->normal[2]);
        if(mag > 1.0E-6){
            aux->normal[0] /= mag;
            aux->normal[1] /= mag;
            aux->normal[2] /= mag;
        }
    }

    float dx = bbox[1][0] - bbox[0][0];
    float dy = bbox[1][1] - bbox[0][1];
    float dz = bbox[1][2] - bbox[0][2];
    float scale = 2.0/ sqrt(dx*dx + dy*dy + dz*dz);

    mesh->scale[0] = mesh->scale[1] = mesh->scale[2] = scale;
    mesh->center[0] = 0.5 * (bbox[1][0] + bbox[0][0]);
    mesh->center[1] = 0.5 * (bbox[1][1] + bbox[0][1]);
    mesh->center[2] = 0.5 * (bbox[1][2] + bbox[0][2]);

    // Check whether read all faces
    if (nfaces != mesh->nfaces) {
        fprintf(stderr, "Se esperaban %d caras, pero se leyeron solo %d caras %s\n", nfaces, mesh->nfaces, filename);
    }

    // Close file
    fclose(fp);

    // Return mesh
    return mesh;
}

void normalize(float * v){
    float magnitude = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    v[0]/=magnitude;
    v[1]/=magnitude;
    v[2]/=magnitude;
}

void getOff(Mesh * mesh, string filename){
    ofstream myfile;
    myfile.open(filename);
    myfile << "OFF" << endl;
    myfile<<mesh->nverts<<" "<<mesh->nfaces<<" "<<0<<endl;

    for(int i = 0; i<mesh->nverts; i++){
        Vertex* aux = &mesh->verts[i];
        myfile<<aux->x<<" "<<aux->y<<" "<<aux->z<<std::endl;
     }

     for(int i = 0; i < mesh->nfaces; i++){
            myfile<<3<<" ";
            Face* aux = &mesh->faces[i];
            for (int j = 0; j< aux->nverts-1; j++){
                myfile<<aux->verts[j]->index<<" ";
            }
            myfile<<aux->verts[aux->nverts-1]->index<<endl;
     }

    myfile.close();
}

bool initResources(const char* filename){
    //Read mesh
    mymesh = readOffFile(filename);

    //Turn mesh into buffers
    vertex_buffer = new float[mymesh->nverts * 3];
    normal_buffer = new float[mymesh->nverts * 3];
    index_buffer = new unsigned int[mymesh->nfaces * 3];
    for(int i = 0; i < mymesh->nverts; i++){
        vertex_buffer[3 * i] = mymesh->verts[i].x;
        vertex_buffer[3 * i + 1] = mymesh->verts[i].y;
        vertex_buffer[3 * i + 2] = mymesh->verts[i].z;

        normal_buffer[3 * i] = mymesh->verts[i].normal[0];
        normal_buffer[3 * i + 1] = mymesh->verts[i].normal[1];
        normal_buffer[3 * i + 2] = mymesh->verts[i].normal[2];
    }
    for(int i = 0; i < mymesh->nfaces; i++){
        Face* aux = &mymesh->faces[i];
        index_buffer[3 * i] = aux->verts[0]->index;
        index_buffer[3 * i + 1] = aux->verts[1]->index;
        index_buffer[3 * i + 2] = aux->verts[2]->index;
    }

    //Send the geometry to the GPU
    glGenBuffers(1, &vbo_vertices);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
    glBufferData(GL_ARRAY_BUFFER, mymesh->nverts * 3 * sizeof(float), vertex_buffer, GL_STATIC_DRAW);

    //Create the index buffer
    glGenBuffers(1, &ibo_triangles);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_triangles);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mymesh->nfaces * 3 * sizeof(unsigned int), index_buffer, GL_STATIC_DRAW);

    //Create the shader
    GLint link_ok = GL_FALSE;
    GLuint vs, fs;
    if((vs = create_shader("depthmap.v.glsl", GL_VERTEX_SHADER))==0) return false;
    if((fs = create_shader("depthmap.f.glsl", GL_FRAGMENT_SHADER))==0) return false;
    program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &link_ok);
    if(!link_ok){
        cout << "Problemas con el linkeo del Shader" << endl;
        print_log(program);
        return false;
    }

    //Link the shader to the program
    attribute_v_coord = glGetAttribLocation(program, "v_coord");
    if(attribute_v_coord == -1){
        cout << "No se puede asociar atributo v_coord" << endl;
        return false;
    }
    uniform_m = glGetUniformLocation(program, "m");
    if(uniform_m == -1){
        cout << "No se puede asociar uniform m" << endl;
        return false;
    }
    uniform_v = glGetUniformLocation(program, "v");
    if(uniform_v == -1){
        cout << "No se puede asociar uniform v" << endl;
        return false;
    }
    uniform_p = glGetUniformLocation(program, "p");
    if(uniform_p == -1){
        cout << "No se puede asociar uniform p" << endl;
        return false;
    }

    return true;
}

void render(float &angleX, float &angleY, float &angleZ, float &angleCam){
    //Transform
    glm::mat4 traslacion = glm::translate(glm::mat4(1.0f), glm::vec3(-mymesh->center[0], -mymesh->center[1], -mymesh->center[2]));
    glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(mymesh->scale[0], mymesh->scale[1], mymesh->scale[2]));
    glm::mat4 rotacionX = glm::rotate(glm::mat4(1.0f), glm::radians(angleX), glm::vec3(1.0f, 0.0f, 0.0f));
    glm::mat4 rotacionY = glm::rotate(glm::mat4(1.0f), glm::radians(angleY), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 rotacionZ = glm::rotate(glm::mat4(1.0f), glm::radians(angleZ), glm::vec3(0.0f, 0.0f, 1.0f));
    glm::mat4 model = rotacionX * rotacionY * rotacionZ * scale * traslacion;

    glm::mat4 view  = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.5f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-sin(angleCam*M_PI/180.0), cos(angleCam*M_PI/180.0), 0.0f));
    glm::mat4 projection = glm::ortho<float>(-1.0,1.0,-1.0,1.0,-1.0,1.0);

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    glUseProgram(program);
    glUniformMatrix4fv(uniform_m, 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(uniform_v, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(uniform_p, 1, GL_FALSE, glm::value_ptr(projection));

    glEnableVertexAttribArray(attribute_v_coord);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
    glVertexAttribPointer(
        attribute_v_coord,
        3,
        GL_FLOAT,
        GL_FALSE,
        0,0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_triangles);
    glDrawElements(GL_TRIANGLES, mymesh->nfaces * 3, GL_UNSIGNED_INT, 0);
    glDisableVertexAttribArray(attribute_v_coord);
    glutSwapBuffers();
}

void freeResources(){
    //Delete Mesh
    delete[] mymesh->verts;
    delete[] mymesh->faces;
    delete mymesh;

    delete[] vertex_buffer;
    delete[] normal_buffer;
    delete[] index_buffer;

    glDeleteProgram(program);
    glDeleteBuffers(1, &vbo_vertices);
    glDeleteBuffers(1, &ibo_triangles);
}

void initGlut(int argc, char *argv[]){
    glutInit(&argc, argv);
    glutInitContextVersion(2,0);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(screen_width, screen_height);
    glutCreateWindow("Offviewer with light");
    GLenum glew_status = glewInit();
    if(glew_status != GLEW_OK){
        cout << "No se puede inicializar GLEW" << endl;
        exit(EXIT_FAILURE);
    }
    if(!GLEW_VERSION_2_0){
        cout << "Tu tarjeta gráfica no soporta OpenGL 4.5" << endl;
        exit(EXIT_FAILURE);
    }
}

void getBitArray(GLubyte* & bits, int & w, int & h){
    GLint viewport[4]; //current viewport
    glGetIntegerv(GL_VIEWPORT, viewport);
    w = viewport[2];
    h = viewport[3];
    bits = new GLubyte[w*3*h];

    glPixelStorei(GL_PACK_ALIGNMENT,1); //or glPixelStorei(GL_PACK_ALIGNMENT,4);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glPixelStorei(GL_PACK_SKIP_ROWS, 0);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
    glReadPixels(0, 0, w, h, GL_BGR_EXT, GL_UNSIGNED_BYTE, bits);
}

void getPgm(GLubyte * bits, int w, int h, int loopInd, float angleCam){
    FILE *fp = NULL;
    std::string filename;
    stringstream ss;
    ss << fixed << setprecision(0) << angleCam;
    filename ="img" + std::to_string(loopInd) + "_" + ss.str() + ".pgm";
    fp = fopen(filename.c_str(), "wb");
    if(fp == NULL){
       cout<<"Cannot create file "<< endl;
       system("PAUSE");
       exit(0);
    }
    char input[10];
    fputs("P5\n",fp);  //Escritura del codigo P5
    sprintf(input,"%d",w);
    fputs(input,fp);//Ancho de la imagen
    fputs("\n",fp);
    sprintf(input,"%d",h);
    fputs(input,fp);//Ancho de la imagen
    fputs("\n",fp);
    sprintf(input,"%d",255);
    fputs(input,fp);//Ancho de la imagen
    fputs("\n",fp);
    int value;
    unsigned char v;
    register unsigned int i,j;
    for(i=0;i<h;i++){
        for(j=0;j<w;j++){
            value = (bits[(h-i-1)*3*w + j*3+0] + bits[(h-i-1)*3*w + j*3+1] + bits[(h-i-1)*3*w + j*3+2])/3;
            v = (unsigned int)value;
            fwrite(&v,sizeof(unsigned char),1,fp);
        }
    }
    fclose(fp);
}

void processImage(int loopInd, float angleX, float angleY, float angleZ, float angleCam){
    
    //Get bit array
    GLubyte * bits; //RGB bits
    int w, h;
    //clock.tick();
    getBitArray(bits, w, h);
    
    //Output PGM file
    getPgm(bits,w,h,loopInd,angleCam);
    
    //Delete bit array
    delete[] bits;
}

////////////////////////////////////////////////////////////
// MAIN PROCEDURE
////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    //Validate parameters
    if(argc==1){
        cout<<"You must provide an OFF file as parameter"<<endl;
        exit(EXIT_FAILURE);
    }
    
    //Initialize random seed
    srand(time(NULL));

    //Initialize GLUT
    initGlut(argc, argv);

    //Initialize resources
    initResources(argv[1]);

    //Loop
    
    for (int i = 0; i < nLoops; i++){
        float angleX = static_cast <float> (rand()%100) / static_cast <float> (100/180.0);
        float angleY = static_cast <float> (rand()%100) / static_cast <float> (100/180.0);
        float angleZ = static_cast <float> (rand()%100) / static_cast <float> (100/180.0);

        cout << "angle: " << angleZ << endl;

        for (float angleCam = 0.0; angleCam < 180.0; angleCam = angleCam + 20.0){
            //Render an image
            
            render(angleX, angleY, angleZ, angleCam);
            //Process the image
            processImage(i, angleX, angleY, angleZ, angleCam);
           
        }
    }

   
    //Free resources
    freeResources();

    return EXIT_SUCCESS;
}
