// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "CommonLib.generated.h"

/**
 * 
 */
UCLASS()
class BITUTILS_API UCommonLib : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

	UFUNCTION(BlueprintCallable, Category = "Timeline_EX")
		static bool AddEventToTimeline(class UTimelineComponent* Timeline, float Time, FName EventName, UObject* InObject);

	UFUNCTION(BlueprintCallable, Category = "Timeline_EX")
		static bool BindFinishEventToTimeline(class UTimelineComponent* Timeline, FName EventName, UObject* InObject);

	UFUNCTION(BlueprintCallable, Category = "Timeline_EX")
		static void UnbindFinishEventToTimeline(class UTimelineComponent* Timeline, UObject* InObject);

	UFUNCTION(BlueprintCallable, Category = "Function_EX")
		static void CallFunctionByName(FName FuncName, UObject* InObject);
	
};
