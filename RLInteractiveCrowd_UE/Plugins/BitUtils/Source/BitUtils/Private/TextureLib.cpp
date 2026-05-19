// Fill out your copyright notice in the Description page of Project Settings.


#include "TextureLib.h"
#include "Engine/Texture2D.h"
#include "Engine/Texture.h"
#include "Engine/Engine.h"
#include "Rendering/Texture2DResource.h"
#include "DynamicRHI.h"


struct GridNode
{
    FVector2D GridID;
    float Value;

    GridNode()
    {
        GridID = FVector2D();
        Value = 0.0f;
    }

    GridNode(FVector2D InGridID, float InValue)
    {
        GridID = InGridID;
        Value = InValue;
    }

    bool operator<(const GridNode Other) const
    {
        return Value < Other.Value;
    }
};

struct GridNodePredict
{
    bool operator() (const GridNode& A, const GridNode& B) const
    {
        // Inverted compared to std::priority_queue - higher priorities float to the top
        return A.Value > B.Value;
    }
};

FVector2D UTextureLib::ExtractTextureData(UTexture2D* Texture, TArray<FColor>& OutPixelData)
{	
    TextureCompressionSettings OldCompressionSettings = Texture->CompressionSettings; TextureMipGenSettings OldMipGenSettings = Texture->MipGenSettings; bool OldSRGB = Texture->SRGB;

    Texture->CompressionSettings = TextureCompressionSettings::TC_VectorDisplacementmap;
    Texture->MipGenSettings = TextureMipGenSettings::TMGS_NoMipmaps;
    Texture->SRGB = false;
    Texture->UpdateResource();

    const FColor* FormatedImageData = static_cast<const FColor*>(Texture->GetPlatformData()->Mips[0].BulkData.LockReadOnly());
    int32 SizeX = Texture->GetPlatformData()->Mips[0].SizeX;
    int32 SizeY = Texture->GetPlatformData()->Mips[0].SizeY;


    OutPixelData.Empty();
    for (int32 Y = 0; Y < SizeY; Y++)
    {
        for (int32 X = 0; X < SizeX; X++)
        {
            OutPixelData.Add(FormatedImageData[Y * SizeX + X]);
        }
    }

    Texture->GetPlatformData()->Mips[0].BulkData.Unlock();

    Texture->CompressionSettings = OldCompressionSettings;
    Texture->MipGenSettings = OldMipGenSettings;
    Texture->SRGB = OldSRGB;
    Texture->UpdateResource();

    return FVector2D(SizeX, SizeY);
}

FColor UTextureLib::GetPixelColorFromStoredData(const TArray<FColor>& PixelData, int32 TextureWidth, int32 TextureHeight, float U, float V)
{
    int32 X = FMath::Clamp(static_cast<int32>(U * TextureWidth), 0, TextureWidth - 1);
    int32 Y = FMath::Clamp(static_cast<int32>(V * TextureHeight), 0, TextureHeight - 1);
    
    return PixelData[Y * TextureWidth + X];
}

TArray<float> UTextureLib::CreateTextureGrid(TArray<FColor> ImageData, int32 ImgSizeX, int32 ImgSizeY, int32 GridSize)
{
    // Convert image data to grayscale
    TArray<uint8> GrayImageData;
    for (const FColor& Pixel : ImageData)
    {
        GrayImageData.Add(FMath::RoundToInt(0.2989 * Pixel.R + 0.5870 * Pixel.G + 0.1140 * Pixel.B)); // Convert to grayscale
    }

    // Create grid
    int32 GridHeight = ImgSizeY / GridSize;
    int32 GridWidth = ImgSizeX / GridSize;
    TArray<float> Grid;

    Grid.Init(0.0f, GridWidth * GridHeight);

    // Fill grid with average grayscale values
    for (int32 i = 0; i < GridHeight; ++i)
    {
        for (int32 j = 0; j < GridWidth; ++j)
        {
            float AverageGrayValue = 0.0f;
            for (int32 y = i * GridSize; y < (i + 1) * GridSize; ++y)
            {
                for (int32 x = j * GridSize; x < (j + 1) * GridSize; ++x)
                {
                    AverageGrayValue += GrayImageData[y * ImgSizeX + x];
                }
            }
            AverageGrayValue /= GridSize * GridSize;
            Grid[i * GridWidth + j] = AverageGrayValue;
        }
    }

    return Grid;
}

TArray<FVector2D> UTextureLib::FindPathInTexture(TArray<float> GridData, int32 GridNumX, int32 GridNumY, FVector2D Start, FVector2D Goal)
{
    TArray<FVector2D> Path;

    // A* Search
    TArray<GridNode> OpenSetTest;
    OpenSetTest.HeapPush(GridNode(Start, 0));


    TMap<FVector2D, FVector2D> CameFrom;
    TMap<FVector2D, float> GScore;
    TMap<FVector2D, float> FScore;

    float inf = 99999999.f;
    for (int y = 0; y < GridNumY; ++y) {
        for (int x = 0; x < GridNumX; ++x) {
            FVector2D position(x, y);
            GScore.Add(position, inf);
            FScore.Add(position, inf);
        }
    }

    GScore[Start] = 0.f;
    FScore[Start] = (Goal - Start).Size(); // Heuristic
    GridNode CurrentNode;
    FVector2D Current;

    //UE_LOG(LogTemp, Log, TEXT("GridNumX = %d, GridNumY = %d,"), GridNumX, GridNumY);

    while (!OpenSetTest.IsEmpty())
    {
        //UE_LOG(LogTemp, Log, TEXT("OpenSet Size = %d"), OpenSetTest.Num());
        OpenSetTest.HeapPop(CurrentNode);
        Current = CurrentNode.GridID;

        if (Current == Goal)
        {
            // Reconstruct path
            while (CameFrom.Contains(Current))
            {
                Path.Insert(Current, 0);
                Current = CameFrom[Current];
            }
            Path.Insert(Start, 0);
            break;
        }

        // Neighbors
        TArray<FVector2D> Neighbors = {
            FVector2D(-1, 0),
            FVector2D(1, 0),
            FVector2D(0, -1),
            FVector2D(0, 1)
        };

        for (const FVector2D& NeighborOffset : Neighbors)
        {
            FVector2D Neighbor = Current + NeighborOffset;
            //UE_LOG(LogTemp, Log, TEXT("Neighbour = (%f, %f)"), Neighbor.X, Neighbor.Y);
            if (Neighbor.X >= 0 && Neighbor.X < GridNumX && Neighbor.Y >= 0 && Neighbor.Y < GridNumY)
            {
                float TentativeGScore = GScore[Current] + 255 - GridData[Neighbor.Y * GridNumX + Neighbor.X];
                //UE_LOG(LogTemp, Log, TEXT("TentativeGScore = %f, GScoreNeighbor = %f"), TentativeGScore, GScore.FindRef(Neighbor));
                if (TentativeGScore < GScore.FindRef(Neighbor))
                {
                    CameFrom.Add(Neighbor, Current);
                    GScore.Add(Neighbor, TentativeGScore);
                    FScore.Add(Neighbor, GScore[Neighbor] + (Goal - Neighbor).Size()); // Heuristic
                    
                    OpenSetTest.HeapPush(GridNode(Neighbor, FScore[Neighbor]));
                    //UE_LOG(LogTemp, Log, TEXT("Score = %f"), FScore[Neighbor]);
                }
            }
        }
    }

    return Path;
}
