#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "obj_loader.h"
//#include <glm/gtc/matrix_inverse.hpp>

using namespace std;
using json = nlohmann::json;

void Scene::loadObj(const std::string& filename, const glm::mat4& transform)
{
    objl::Loader loader;
    if (!loader.LoadFile(filename)) {
        std::cerr << "[Scene] Failed to load OBJ: " << filename << std::endl;
        return;
    }

    // textures are referenced relative to the .obj's folder
    std::string dir = filename.substr(0, filename.find_last_of("/\\") + 1);
    glm::mat3 normalMatrix = glm::inverseTranspose(glm::mat3(transform));

    for (const objl::Mesh& mesh : loader.LoadedMeshes)
    {
        // build material from mtl
        
        MaterialHost mat{};
        mat.name = mesh.MeshMaterial.name;
        mat.color = glm::vec3(mesh.MeshMaterial.Kd.X,
            mesh.MeshMaterial.Kd.Y,
            mesh.MeshMaterial.Kd.Z);
        if (mat.color == glm::vec3(0.0f)) mat.color = glm::vec3(1.0f);
        mat.shininess = (mesh.MeshMaterial.Ns > 0.0f)
            ? mesh.MeshMaterial.Ns : 0.0f;
        if (!mesh.MeshMaterial.map_Kd.empty())
        {
            mat.diffuseTexPath = mesh.MeshMaterial.map_Kd;
            std::cout << mesh.MeshMaterial.map_Kd
                << "\n";
        }
        glm::vec3 emission = { mesh.MeshMaterial.Ke.X,  mesh.MeshMaterial.Ke.Y, mesh.MeshMaterial.Ke.Z };
        if (emission.length() > 0.0f)
            mat.emittance = emission.x;
        if (!mesh.MeshMaterial.map_Ks.empty())  
            mat.specularTexPath = dir + mesh.MeshMaterial.map_Ks;
        else
            mat.roughness = mesh.MeshMaterial.Ns;
        materials.push_back(mat);
        

        // Every consecutive triple of indices is one triangle
        for (size_t i = 0; i + 2 < mesh.Indices.size(); i += 3)
        {
            Triangle tri{};
            tri.materialid = materials.size()-1;
            for (int k = 0; k < 3; ++k) {
                const objl::Vertex& vtx = mesh.Vertices[mesh.Indices[i + k]];
                glm::vec3 pos(vtx.Position.X, vtx.Position.Y, vtx.Position.Z);
                glm::vec3 nor(vtx.Normal.X, vtx.Normal.Y, vtx.Normal.Z);
                tri.v[k] = glm::vec3(transform * glm::vec4(pos, 1.0f));
                tri.n[k] = nor;                       // transformed below
                tri.uv[k] = glm::vec2(vtx.TextureCoordinate.X,
                    1.0f - vtx.TextureCoordinate.Y);
            }
            // bake normals to world space; fall back to face normal if missing
            glm::vec3 faceN = glm::normalize(
                glm::cross(tri.v[1] - tri.v[0], tri.v[2] - tri.v[0]));
            for (int k = 0; k < 3; ++k)
                tri.n[k] = (glm::length(tri.n[k]) < 1e-6f)
                ? faceN : glm::normalize(normalMatrix * tri.n[k]);

            triangles.push_back(tri);
        }
    }
    std::cout << "[Scene] " << filename << " -> "
        << triangles.size() << " total triangles\n";
}



Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        MaterialHost newMaterial{};
   
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.shininess = 1.0f - p["ROUGHNESS"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "mesh")
        {
            glm::vec3 t(p["TRANS"][0], p["TRANS"][1], p["TRANS"][2]);
            glm::vec3 r(p["ROTAT"][0], p["ROTAT"][1], p["ROTAT"][2]);
            glm::vec3 s(p["SCALE"][0], p["SCALE"][1], p["SCALE"][2]);
            glm::mat4 m = utilityCore::buildTransformationMatrix(t, r, s);

  
            loadObj(p["FILE"].get<std::string>(), m);
            continue;
        }
        
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
