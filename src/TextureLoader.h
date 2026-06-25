#pragma once
#include <string>
#include <cuda_runtime.h>
#include <stb_image.h>
#include <iostream>
struct CUDATexture
{
    cudaArray_t array = nullptr;
    cudaTextureObject_t object = 0;

    int width = 0;
    int height = 0;

    bool valid() const
    {
        return object != 0;
    }

    void destroy();
};

class TextureLoader
{
public:

    static CUDATexture loadTexture(const std::string& filename)
    {

        int w, h, channels;

        unsigned char* pixels = stbi_load( filename.c_str(),  &w,&h, &channels, STBI_rgb_alpha);

        if (!pixels)
        {
            std::cout << "found nothing at" << filename << "\n";
            return {};
        }


        cudaChannelFormatDesc channel = cudaCreateChannelDesc<uchar4>();

        cudaArray_t array;

        cudaMallocArray(  &array,  &channel, w,h);


        cudaMemcpy2DToArray( array, 0,  0,  pixels, w * sizeof(uchar4),  w * sizeof(uchar4), h, cudaMemcpyHostToDevice);

        stbi_image_free(pixels);

        cudaResourceDesc res{};

        res.resType = cudaResourceTypeArray;
        res.res.array.array = array;

        cudaTextureDesc tex{};

        tex.addressMode[0] = cudaAddressModeWrap;
        tex.addressMode[1] = cudaAddressModeWrap;

        tex.filterMode = cudaFilterModeLinear;

        tex.readMode = cudaReadModeNormalizedFloat;

        tex.normalizedCoords = 1;


        cudaTextureObject_t object;

        cudaCreateTextureObject(  &object,  &res, &tex,  nullptr);


        CUDATexture texture;

        texture.array = array;
        texture.object = object;

        texture.width = w;
        texture.height = h;

        return texture;

    }



    static CUDATexture loadMapKd(const std::string& textureDir,
        const std::string& mapKd,
        std::string* resolvedPath = nullptr) {
        if (mapKd.empty()) return {};

        std::string fullPath = textureDir;
        if (!fullPath.empty() && fullPath.back() != '/' && fullPath.back() != '\\')
            fullPath += "/";
        fullPath += mapKd;

        CUDATexture tex = loadTexture(fullPath);
        if (tex.object) {  
            if (resolvedPath) *resolvedPath = fullPath;
            std::cout << "[TextureLoader] Loaded '" << mapKd << "' from '" << fullPath << "'\n";
            return tex;  
        }

        std::cerr << "[TextureLoader] Could not find '" << mapKd << "' in '" << fullPath << "'\n";
        return {};
    }
}; 
