// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "AgentBase.h"
#include "CustomStruct.h"
#include "Kismet/KismetSystemLibrary.h"
#include "AgentRL.generated.h"

/**
 * 
 */
UCLASS()
class RLINTERACTIVECROWD_API AAgentRL : public AAgentBase
{
	GENERATED_BODY()

	AAgentRL(const FObjectInitializer& ObjectInitializer);

public:
	UPROPERTY(BlueprintReadWrite, Category = "SenseSetting")
		int32 num_rays;
	UPROPERTY(BlueprintReadWrite, Category = "SenseSetting")
		float max_ray_distance;
	UPROPERTY(BlueprintReadWrite, Category = "SenseSetting")
		float ray_vision_lamda;
	UPROPERTY(BlueprintReadWrite, Category = "SenseSetting")
		float max_time_overlaped_with_obstacle;
	UPROPERTY(BlueprintReadWrite, Category = "SenseSetting")
		float arrive_range_scale;

	UPROPERTY(BlueprintReadWrite, Category = "Sense")
		TArray<float> ray_distances;
	UPROPERTY(BlueprintReadWrite, Category = "Sense")
		TArray<float> max_ray_distances;
	UPROPERTY(BlueprintReadWrite, Category = "Sense")
		TArray<FVector2D> ray_object_relative_velocities;
	UPROPERTY(BlueprintReadWrite, Category = "Sense")
		TArray<AAgentBase*> collision_agents;
	UPROPERTY(BlueprintReadWrite, Category = "Sense")
		TArray<float> collision_distances; 
	UPROPERTY(BlueprintReadWrite, Category = "Sense")
		bool is_overlaped_with_obstacle;
	UPROPERTY(BlueprintReadWrite, Category = "Sense")
		float overlap_obstacle_time;
	UPROPERTY(BlueprintReadWrite, Category = "Sense")
		TEnumAsByte<EDrawDebugTrace::Type> draw_trace_type;

	UPROPERTY(BlueprintReadWrite, Category = "ImplicitEnv")
		FVector2D implicit_env_obs_range;
	UPROPERTY(BlueprintReadWrite, Category = "ImplicitEnv")
		float implicit_env_value;
	UPROPERTY(BlueprintReadWrite, Category = "ImplicitEnv")
		float implicit_env_value_threshold;

	UPROPERTY(BlueprintReadWrite, Category = "Nav")
		bool use_nav_point;
	UPROPERTY(BlueprintReadWrite, Category = "Nav")
		TArray<FVector2D> nav_points;
	UPROPERTY(BlueprintReadWrite, Category = "Nav")
		int32 current_nav_point_index;
	UPROPERTY(BlueprintReadWrite, Category = "Nav")
		float update_nav_point_threshold;
	UPROPERTY(BlueprintReadWrite, Category = "Nav")
		float nav_point_approach_diatance;

	UPROPERTY(BlueprintReadWrite, Category = "Reward")
		FS_RewardConfig reward_config;
	UPROPERTY(BlueprintReadWrite, Category = "Reward")
		float reward_weight_entity;
	UPROPERTY(BlueprintReadWrite, Category = "Reward")
		float reward_weight_implicit;
	UPROPERTY(BlueprintReadWrite, Category = "Reward")
		float reward_cur_step;
	UPROPERTY(BlueprintReadWrite, Category = "Reward")
		float reward_cnt;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Default")
		class AEnvBase* env;

public:
	UFUNCTION(BlueprintCallable, Category = "Default")
		void init_agent_rl(FS_AgentConfig agent_config);

	UFUNCTION(BlueprintCallable, Category = "RL")
		void update(TArray<float> action);

	UFUNCTION(BlueprintCallable, Category = "RL")
		void update_without_action();

	UFUNCTION(BlueprintCallable, Category = "RL")
		void sense();

	UFUNCTION(BlueprintCallable, Category = "RL")
		FS_FloatArray get_observation();

	UFUNCTION(BlueprintCallable, Category = "RL")
		float get_reward();

	UFUNCTION(BlueprintCallable, Category = "RL")
		bool is_done();

	UFUNCTION(BlueprintCallable, Category = "RL")
		bool is_arrived_target();

	// Overlap begin function
	UFUNCTION()
		void OnOverlapBegin(UPrimitiveComponent* OverlappedComp, AActor* OtherActor,
			UPrimitiveComponent* OtherComp, int32 OtherBodyIndex,
			bool bFromSweep, const FHitResult& SweepResult);

	// Overlap end function
	UFUNCTION()
		void OnOverlapEnd(UPrimitiveComponent* OverlappedComp, AActor* OtherActor,
			UPrimitiveComponent* OtherComp, int32 OtherBodyIndex);

	virtual void UpdateVisualization_Implementation();
	
};
