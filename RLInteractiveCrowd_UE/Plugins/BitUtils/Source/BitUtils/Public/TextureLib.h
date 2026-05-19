// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "TextureLib.generated.h"

/**
 * 
 */
UCLASS()
class BITUTILS_API UTextureLib : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintCallable, Category = "Texture_EX")
		static FVector2D ExtractTextureData(class UTexture2D* Texture, TArray<FColor>& OutPixelData);

	UFUNCTION(BlueprintCallable, Category = "Texture_EX")
		static FColor GetPixelColorFromStoredData(const TArray<FColor>& PixelData, int32 TextureWidth, int32 TextureHeight, float U, float V);

	UFUNCTION(BlueprintCallable, Category = "Texture_EX")
		static TArray<float> CreateTextureGrid(TArray<FColor> PixelData, int32 ImgSizeX, int32 ImgSizeY, int32 GridSize);

	UFUNCTION(BlueprintCallable, Category = "Texture_EX")
		static TArray<FVector2D> FindPathInTexture(TArray<float> GridData, int32 GridNumX, int32 GridNumY, FVector2D Start, FVector2D Goal);

};
