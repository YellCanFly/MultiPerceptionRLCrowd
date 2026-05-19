// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "AgentBase.generated.h"

UCLASS()
class RLINTERACTIVECROWD_API AAgentBase : public AActor
{
	GENERATED_BODY()

public:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
		class USceneComponent* Root;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
		class USphereComponent* SphereCollision;

	UPROPERTY(BlueprintReadWrite, Category = "BasicSetting")
		int32 index;
	UPROPERTY(BlueprintReadWrite, Category = "BasicSetting")
		float radius;
	UPROPERTY(BlueprintReadWrite, Category = "BasicSetting")
		float max_alive_time;
	UPROPERTY(BlueprintReadWrite, Category = "BasicSetting")
		float max_speed_orientation;
	UPROPERTY(BlueprintReadWrite, Category = "BasicSetting")
		float back_forward_velocity_ratiol;
	UPROPERTY(BlueprintReadWrite, Category = "BasicSetting")
		float max_speed_perpendicular;
	UPROPERTY(BlueprintReadWrite, Category = "BasicSetting")
		float max_angular_velocity;
	UPROPERTY(BlueprintReadWrite, Category = "BasicSetting")
		float max_acceleration;

	UPROPERTY(BlueprintReadWrite, Category = "BasicState")
		FVector2D position;
	UPROPERTY(BlueprintReadWrite, Category = "BasicState")
		FVector2D target_position;
	UPROPERTY(BlueprintReadWrite, Category = "BasicState")
		FVector2D position_last_step;
	UPROPERTY(BlueprintReadWrite, Category = "BasicState")
		float orientation;
	UPROPERTY(BlueprintReadWrite, Category = "BasicState")
		FVector2D velocity;
	UPROPERTY(BlueprintReadWrite, Category = "BasicState")
		FVector2D velocity_last_step;
	UPROPERTY(BlueprintReadWrite, Category = "BasicState")
		FVector2D acceleration;
	UPROPERTY(BlueprintReadWrite, Category = "BasicState")
		float angular_velocity;
	UPROPERTY(BlueprintReadWrite, Category = "BasicState")
		float dt;
	UPROPERTY(BlueprintReadWrite, Category = "BasicState")
		float alive_time;
	UPROPERTY(BlueprintReadWrite, Category = "BasicState")
		float agent_to_target_distance;
	UPROPERTY(BlueprintReadWrite, Category = "BasicState")
		float target_approach_distance;
	UPROPERTY(BlueprintReadWrite, Category = "BasicState")
		float target_offset;
	UPROPERTY(BlueprintReadWrite, Category = "BasicState")
		bool bDone;

	
public:	
	// Sets default values for this actor's properties
	AAgentBase();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	UFUNCTION(BlueprintPure, Category = "CoordinateSystem")
		FVector2D velocity_to_absolute(FVector2D relative_velocity) const;

	UFUNCTION(BlueprintPure, Category = "CoordinateSystem")
		FVector2D velocity_to_relative(FVector2D absolute_velocity) const;

	UFUNCTION(BlueprintPure, Category = "CoordinateSystem")
		FVector2D position_to_relative(FVector2D relative_position) const;

	UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "Visualization")
		void UpdateVisualization();
		virtual void UpdateVisualization_Implementation();

	UFUNCTION(BlueprintCallable, Category = "SetState")
		void set_agent_location();

};
