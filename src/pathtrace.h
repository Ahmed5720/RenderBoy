#pragma once

#include "scene.h"
#include "utilities.h"
#include "TextureLoader.h"
void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
