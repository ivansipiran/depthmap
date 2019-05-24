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
/*
void prepareDescriptors(GLubyte* bits, int w, int h, vector<WindowElement>& window, vector<Descriptor>& descriptors){
    //Prepare candidates
    vector<Descriptor> candidates;
    candidates.clear();
    for (int i = window_width/2; i< screen_width - window_width/2 -1; i++){
        for (int j = window_height/2; j<screen_height - window_height/2 - 1; j++){
            if (getValueFromBitArray(bits,w,h,i,j) > 0.1){
                candidates.push_back(Descriptor(i,j));
            }
        }
    }

    descriptors.clear();
    for (int i = 0; i < num_seeds; i++){
        //Pick a candidate
        int index = rand() % candidates.size();
        Descriptor descriptor = candidates.at(index);
        candidates.erase(candidates.begin()+index);
        descriptor.setDescriptor(0);

        //Build window
        for (int j=0; j<window.size();j++){
            int x1 = descriptor.getX() + window.at(j).getX1() - window_width/2;
            int x2 = descriptor.getX() + window.at(j).getX2() - window_width/2;
            int y1 = descriptor.getY() + window.at(j).getY1() - window_height/2;
            int y2 = descriptor.getY() + window.at(j).getY2() - window_height/2;
            if (getValueFromBitArray(bits,w,h,x1,y1) > getValueFromBitArray(bits,w,h,x2,y2)){
                descriptor.setDescriptor(descriptor.getDescriptor()+(((unsigned int)1)<<j));
            }
        }

        descriptors.push_back(descriptor);
    }
}

void prepareWindow(vector<WindowElement> &window){
    //Prepare candidates
    vector<WindowElement> candidates;
    for (int i = 0; i < window_height; i++){
        for (int j = 0; j< window_width; j++){
            for (int k = 0; k < window_height; k++){
                for (int l = 0; l< window_width; l++){
                    if (i!=k && j!=l){
                        candidates.push_back(WindowElement(i,j,k,l));
                    }
                }
            }
        }
    }

    //Build window
    window.clear();
    for (int i = 0; i< descriptor_size; i++){
        int index = rand() % candidates.size();
        window.push_back(candidates.at(index));
        candidates.erase(candidates.begin()+index);
    }
}

void compareDescriptors(std::vector<Descriptor>& descriptors1,std::vector<Descriptor>& descriptors2, int axis, int& pairCount,float& linePosition){
    linePosition = 0.0;
    pairCount = 0;
    for (int i = 0; i< descriptors1.size(); i++){
        for (int j = 0; j< descriptors2.size(); j++){
            unsigned int ixor = descriptors1.at(i).getDescriptor() ^ descriptors2.at(j).getDescriptor();
            unsigned int nDiffBits = 0;
            while(ixor){
                nDiffBits += ixor & 1;
                ixor >>= 1;
            }
            if (nDiffBits < bit_diff_threshold){
                if (axis){
                    linePosition += ((float)(descriptors1.at(i).getY() + descriptors2.at(j).getY()))/2.0;
                }
                else{
                    linePosition += ((float)(descriptors1.at(i).getX() + descriptors2.at(j).getX()))/2.0;
                }
                pairCount++;
                j=descriptors2.size();
            }
        }
    }
    linePosition /= ((float)pairCount);
}

void printLineOfSymmetry(GLubyte* bits,int w,int h,int axis,float linePosition){
     if (axis){
         int y = (int) linePosition;
         for (int i = 0; i< screen_width; i++){
            bits[(h-y-1)*3*w + i*3+0] = 255;
            bits[(h-y-1)*3*w + i*3+1] = 255;
            bits[(h-y-1)*3*w + i*3+2] = 255;
         }
     }
     else{
         int x = (int) linePosition;
         for (int i = 0; i< screen_height; i++){
            bits[(h-i-1)*3*w + x*3+0] = 255;
            bits[(h-i-1)*3*w + x*3+1] = 255;
            bits[(h-i-1)*3*w + x*3+2] = 255;
         }
     }
}

void findLinesOfSymmetry(GLubyte* bits, int w, int h, float &xLinePosition, int &xPairCount, float &yLinePosition, int &yPairCount){
    //Prepare window
    vector<WindowElement> window;
    prepareWindow(window);

    //Get descriptors for the current image
    std::vector<Descriptor> descriptors;
    prepareDescriptors(bits,w,h,window,descriptors);

    //Invert window (x axis)
    vector<WindowElement> xWindow;
    for (int i = 0; i< window.size(); i++){
        xWindow.push_back(WindowElement(window_width-(window.at(i).getX1()+1),window.at(i).getY1(),window_width-(window.at(i).getX2()+1),window.at(i).getY2()));
    }

    //Get descriptors (x axis)
    std::vector<Descriptor> xDescriptors;
    prepareDescriptors(bits,w,h,xWindow,xDescriptors);

    //Compare descriptors (x axis)
    xPairCount = 0;
    xLinePosition = 0.0;
    compareDescriptors(descriptors,xDescriptors,0,xPairCount,xLinePosition);

    //Print line of symmetry in bit array
    if (xPairCount > pair_count_threshold){
        printLineOfSymmetry(bits,w,h,0,xLinePosition);
    }

    //Invert window (y axis)
    vector<WindowElement> yWindow;
    for (int i = 0; i< window.size(); i++){
        yWindow.push_back(WindowElement(window.at(i).getX1(),window_height-(window.at(i).getY1()+1),window.at(i).getX2(),window_height-(window.at(i).getY2()+1)));
    }

    //Get descriptors (y axis)
    std::vector<Descriptor> yDescriptors;
    prepareDescriptors(bits,w,h,yWindow,yDescriptors);

    //Compare descriptors (y axis)
    yPairCount = 0;
    yLinePosition = 0.0;
    compareDescriptors(descriptors,yDescriptors,1,yPairCount,yLinePosition);

    //Print line of symmetry in bit array
    if (yPairCount > pair_count_threshold){
        printLineOfSymmetry(bits,w,h,1,yLinePosition);
    }
}*/

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
/*
Mesh * createPlaneFromCoefficients(float * coefficients){
    //Initialize mesh
    Mesh * mesh = new Mesh();
    mesh->nverts = 4;
    mesh->nfaces = 2;
    mesh->scale[0] = mymesh->scale[0];
    mesh->scale[1] = mymesh->scale[1];
    mesh->scale[2] = mymesh->scale[2];

    //Find center of mesh
    float distance = (coefficients[0]*mymesh->center[0] + coefficients[1]*mymesh->center[1] + coefficients[2]*mymesh->center[2] + coefficients[3]);
    mesh->center[0] = mymesh->center[0] + distance*coefficients[0];
    mesh->center[1] = mymesh->center[1] + distance*coefficients[1];
    mesh->center[2] = mymesh->center[2] + distance*coefficients[2];

    //Find two perpendicular vector
    float v1[3];
    float v2[3];

    if (coefficients[0]==0){
        if (coefficients[1]==0){
            if (coefficients[2]==0){
                //Should never get here
                v1[0] = v1[1] = v1[2] = v2[0] = v2[1] = v2[2] = 0.0;
            }
            else{
                //X axis and Y axis
                v1[0] = 1.0;
                v1[1] = 0.0;
                v1[2] = 0.0;
                v2[0] = 0.0;
                v2[1] = 1.0;
                v2[2] = 0.0;
            }
        }
        else{
            if (coefficients[2]==0){
                //X axis and Z axis
                v1[0] = 1.0;
                v1[1] = 0.0;
                v1[2] = 0.0;
                v2[0] = 0.0;
                v2[1] = 0.0;
                v2[2] = 1.0;
            }
            else{
                //X axis and YZ plane
                v1[0] = 1.0;
                v1[1] = 0.0;
                v1[2] = 0.0;
                v2[0] = 0.0;
                v2[1] = coefficients[2];
                v2[2] = -coefficients[1];
                normalize(v2);
            }
        }
    }
    else{
        if (coefficients[1]==0){
            if (coefficients[2]==0){
                //Y axis and Z axis
                v1[0] = 0.0;
                v1[1] = 1.0;
                v1[2] = 0.0;
                v2[0] = 0.0;
                v2[1] = 0.0;
                v2[2] = 1.0;
            }
            else{
                //Y axis and XZ plane
                v1[0] = 0.0;
                v1[1] = 1.0;
                v1[2] = 0.0;
                v2[0] = coefficients[2];
                v2[1] = 0.0;
                v2[2] = -coefficients[0];
                normalize(v2);
            }
        }
        else{
            if (coefficients[2]==0){
                //Z axis and XY plane
                v1[0] = 0.0;
                v1[1] = 0.0;
                v1[2] = 1.0;
                v2[0] = coefficients[1];
                v2[1] = -coefficients[0];
                v2[2] = 0.0;
                normalize(v2);
            }
            else{
                //Could be any combination (YZ and XZ planes)
                v1[0] = 0.0;
                v1[1] = coefficients[2];
                v1[2] = -coefficients[1];
                v2[0] = coefficients[1]*coefficients[1] + coefficients[2]*coefficients[2];
                v2[1] = -coefficients[0]*coefficients[1];
                v2[2] = -coefficients[0]*coefficients[2];
                normalize(v1);
                normalize(v2);
            }
        }
    }

    //Separate memory
    mesh->verts = new Vertex [mesh->nverts];
    assert(mesh->verts);
    mesh->faces = new Face [mesh->nfaces];
    assert(mesh->faces);

    //Create auxiliary variables
    float v1Displacement = 1/mymesh->scale[0];
    float v2Displacement = 1/mymesh->scale[1];

    //Create vertices
    Vertex* vert = &mesh->verts[0];
    vert->index = 0;
    vert->normal[0] = 0.0;
    vert->normal[1] = 0.0;
    vert->normal[2] = 0.0;
    vert->x = mesh->center[0] + v1Displacement*v1[0] + v2Displacement*v2[0];
    vert->y = mesh->center[1] + v1Displacement*v1[1] + v2Displacement*v2[1];
    vert->z = mesh->center[2] + v1Displacement*v1[2] + v2Displacement*v2[2];
    vert->num_faces = 1;

    vert = &mesh->verts[1];
    vert->index = 1;
    vert->normal[0] = coefficients[0];
    vert->normal[1] = coefficients[1];
    vert->normal[2] = coefficients[2];
    vert->x = mesh->center[0] - v1Displacement*v1[0] + v2Displacement*v2[0];
    vert->y = mesh->center[1] - v1Displacement*v1[1] + v2Displacement*v2[1];
    vert->z = mesh->center[2] - v1Displacement*v1[2] + v2Displacement*v2[2];
    vert->num_faces = 2;

    vert = &mesh->verts[2];
    vert->index = 2;
    vert->normal[0] = coefficients[0];
    vert->normal[1] = coefficients[1];
    vert->normal[2] = coefficients[2];
    vert->x = mesh->center[0] - v1Displacement*v1[0] - v2Displacement*v2[0];
    vert->y = mesh->center[1] - v1Displacement*v1[1] - v2Displacement*v2[1];
    vert->z = mesh->center[2] - v1Displacement*v1[2] - v2Displacement*v2[2];
    vert->num_faces = 2;

    vert = &mesh->verts[3];
    vert->index = 3;
    vert->normal[0] = 0.0;
    vert->normal[1] = 0.0;
    vert->normal[2] = 0.0;
    vert->x = mesh->center[0] + v1Displacement*v1[0] - v2Displacement*v2[0];
    vert->y = mesh->center[1] + v1Displacement*v1[1] - v2Displacement*v2[1];
    vert->z = mesh->center[2] + v1Displacement*v1[2] - v2Displacement*v2[2];
    vert->num_faces = 1;

    //Create faces
    Face *face = &mesh->faces[0];
    face->nverts = 3;
    face->verts = new Vertex *[3];
    assert(face->verts);
    face->verts[0] = &(mesh->verts[0]);
    face->verts[1] = &(mesh->verts[1]);
    face->verts[2] = &(mesh->verts[3]);
    face->normal[0] = coefficients[0];
    face->normal[1] = coefficients[1];
    face->normal[2] = coefficients[2];

    face = &mesh->faces[1];
    face->nverts = 3;
    face->verts = new Vertex *[3];
    assert(face->verts);
    face->verts[0] = &(mesh->verts[1]);
    face->verts[1] = &(mesh->verts[2]);
    face->verts[2] = &(mesh->verts[3]);
    face->normal[0] = coefficients[0];
    face->normal[1] = coefficients[1];
    face->normal[2] = coefficients[2];

    return mesh;
}*/

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
/*
void createPlane(float * coefficients){
    //Create plane mesh
    Mesh * plane = createPlaneFromCoefficients(coefficients);

    //Output mesh to file
    string filename = "plane" + std::to_string(planeCount) + ".off";
    getOff(plane,filename);
    planeCount++;

    //Delete mesh
    delete[] plane->verts;
    delete[] plane->faces;
    delete plane;
}
*/

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
//    glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(-mymesh->center[0], -mymesh->center[1], -mymesh->center[2]));
//    model = glm::scale(model, glm::vec3(mymesh->scale[0], mymesh->scale[1], mymesh->scale[2]));
//    model = glm::rotate(model, glm::radians(angleX), glm::vec3(1.0f, 0.0f, 0.0f));
//    model = glm::rotate(model, glm::radians(angleY), glm::vec3(0.0f, 1.0f, 0.0f));
//    model = glm::rotate(model, glm::radians(angleZ), glm::vec3(0.0f, 0.0f, 1.0f));

//    glm::mat4 view  = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.5f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
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
//    glFinish(); //finish all commands of OpenGL
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

/*
void rotateVector(float *normal, float angle, int axis){
    float x = normal[0];
    float y = normal[1];
    float z = normal[2];
    if (axis==0){
        normal[0] = x;
        normal[1] = y*cos(angle*M_PI/180.0) - z*sin(angle*M_PI/180.0);
        normal[2] = y*sin(angle*M_PI/180.0) + z*cos(angle*M_PI/180.0);
    }
    else{
        if (axis==1){
            normal[0] = x*cos(angle*M_PI/180.0) + z*sin(angle*M_PI/180.0);
            normal[1] = y;
            normal[2] = -x*sin(angle*M_PI/180.0) + z*cos(angle*M_PI/180.0);
        }
        else{
            normal[0] = x*cos(angle*M_PI/180.0) - y*sin(angle*M_PI/180.0);
            normal[1] = x*sin(angle*M_PI/180.0) + y*cos(angle*M_PI/180.0);
            normal[2] = z;
        }
    }
    normalize(normal);
}

void findPlanesOfSymmetry(float xLinePosition,int xPairCount,float yLinePosition,int yPairCount,float angleX,float angleY,float angleZ,float angleCam,ofstream &outputFile){
    //Evaluate x
    if (xPairCount > pair_count_threshold || yPairCount > pair_count_threshold){ 
        if (xPairCount > pair_count_threshold){
            //Vertical plane
            float verPlaneEq[4];
            verPlaneEq[0] = 1.0;
            verPlaneEq[1] = 0.0;
            verPlaneEq[2] = 0.0;

            //Rotate camera
            rotateVector(verPlaneEq,angleCam,2);

            //Reverse rotations
            rotateVector(verPlaneEq,-angleX,0);
            rotateVector(verPlaneEq,-angleY,1);
            rotateVector(verPlaneEq,-angleZ,2);

            float distanceFromCenter = (xLinePosition - screen_width/2)/(screen_width*mymesh->scale[0]);
            float pointInPlane[3];
            pointInPlane[0] = mymesh->center[0] + verPlaneEq[0]*distanceFromCenter;
            pointInPlane[1] = mymesh->center[1] + verPlaneEq[1]*distanceFromCenter;
            pointInPlane[2] = mymesh->center[2] + verPlaneEq[2]*distanceFromCenter;

            verPlaneEq[3] = -pointInPlane[0]*verPlaneEq[0] - pointInPlane[1]*verPlaneEq[1] - pointInPlane[2]*verPlaneEq[2];
            createPlane(verPlaneEq);
            outputFile <<verPlaneEq[0] << " " << verPlaneEq[1] << " " << verPlaneEq[2] << " " << verPlaneEq[3] << "\n";
        }

        if (yPairCount > pair_count_threshold){
            //Horizontal plane
            float horPlaneEq[4];
            horPlaneEq[0] = 0.0;
            horPlaneEq[1] = 1.0;
            horPlaneEq[2] = 0.0;

            //Apply camara transformation
            rotateVector(horPlaneEq,angleCam,2);

            //Reverse rotations
            rotateVector(horPlaneEq,-angleX,0);
            rotateVector(horPlaneEq,-angleY,1);
            rotateVector(horPlaneEq,-angleZ,2);

            float distanceFromCenter = (yLinePosition - screen_height/2)/(screen_height*mymesh->scale[0]);
            float pointInPlane[3];
            pointInPlane[0] = mymesh->center[0] + horPlaneEq[0]*distanceFromCenter;
            pointInPlane[1] = mymesh->center[1] + horPlaneEq[1]*distanceFromCenter;
            pointInPlane[2] = mymesh->center[2] + horPlaneEq[2]*distanceFromCenter;

            horPlaneEq[3] = -pointInPlane[0]*horPlaneEq[0] - pointInPlane[1]*horPlaneEq[1] - pointInPlane[2]*horPlaneEq[2];
            createPlane(horPlaneEq);
            outputFile <<horPlaneEq[0] << " " << horPlaneEq[1] << " " << horPlaneEq[2] << " " << horPlaneEq[3] << "\n";
        }
    }
}*/

void processImage(int loopInd, float angleX, float angleY, float angleZ, float angleCam){
    //Timer
    //Util::Clock clock;

    //Get bit array
    GLubyte * bits; //RGB bits
    int w, h;
    //clock.tick();
    getBitArray(bits, w, h);
    //clock.tick();
    //cout << "Generate bit array time:" << clock.getTime() << endl;

//    getPgm(bits,w,h,loopInd,0.0);

    //Find lines of symmetry
    //float xLinePosition, yLinePosition;
    //int xPairCount, yPairCount;

    //findLinesOfSymmetry(bits,w,h,xLinePosition,xPairCount,yLinePosition,yPairCount);
    //clock.tick();
    //cout << "Find line of symmetry time:" << clock.getTime() << endl;

    //Find planes of symmetry
    //findPlanesOfSymmetry(xLinePosition,xPairCount,yLinePosition,yPairCount,angleX,angleY,angleZ,angleCam,outputFile);
    //clock.tick();
    //cout << "Find plane of symmetry time:" << clock.getTime() << endl;

    //Output PGM file
    getPgm(bits,w,h,loopInd,angleCam);
    //clock.tick();
    //cout << "Generate pgm time:" << clock.getTime() << endl;

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
    /*fs::path full_path = fs::system_complete( fs::path( argv[1] ) );
    if(!fs::exists( full_path ) || fs::is_directory( full_path ) ){
        cout<<"File not found"<<endl;
        exit(EXIT_FAILURE);
    }*/

    //Initialize random seed
    srand(time(NULL));

    //Initialize GLUT
    initGlut(argc, argv);

    //Initialize resources
    initResources(argv[1]);

    //Create output file
    //ofstream outputFile;
    //outputFile.open("output.txt");

    //Loop
    
    for (int i = 0; i < nLoops; i++){
        float angleX = static_cast <float> (rand()%100) / static_cast <float> (100/180.0);
        float angleY = static_cast <float> (rand()%100) / static_cast <float> (100/180.0);
        float angleZ = static_cast <float> (rand()%100) / static_cast <float> (100/180.0);

        cout << "angle: " << angleZ << endl;

        for (float angleCam = 0.0; angleCam < 180.0; angleCam = angleCam + 20.0){
            //Render an image
            
            render(angleX, angleY, angleZ, angleCam);
            
//            cout << "Rendering time:" << clock.getTime() << "\n";

            //Process the image
            
            processImage(i, angleX, angleY, angleZ, angleCam);
           
//            cout << "Processing time:" << clock.getTime() << "\n";
        }
    }

    //Close output file
    //outputFile.close();

    //Free resources
    freeResources();

    return EXIT_SUCCESS;
}
