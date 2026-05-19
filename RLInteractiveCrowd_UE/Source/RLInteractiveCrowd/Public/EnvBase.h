// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "CustomStruct.h"
#include "EnvBase.generated.h"

UCLASS()
class RLINTERACTIVECROWD_API AEnvBase : public AActor
{
	GENERATED_BODY()
	
public:
	UPROPERTY(BlueprintReadWrite, Category = "BasicSetting")
		FS_EnvConfig env_config;
	UPROPERTY(BlueprintReadWrite, Category = "BasicSetting")
		FS_RewardConfig reward_config;
	UPROPERTY(BlueprintReadWrite, Category = "BasicSetting")
		FVector2D env_size;
	UPROPERTY(BlueprintReadWrite, Category = "BasicSetting")
		int32 num_agents;
	UPROPERTY(BlueprintReadWrite, Category = "BasicSetting")
		int32 num_obstacles;

	UPROPERTY(BlueprintReadWrite, Category = "Default")
		TSubclassOf<class AAgentRL> AgentRLClass;
	UPROPERTY(BlueprintReadWrite, Category = "Default")
		TArray<class AAgentRL*> agents;
		TArray<class AAgentRL*> new_agents;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Default")
		TArray<TSubclassOf<AActor>> InitPosCollisionCheckClasses;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Default")
		TArray<TSubclassOf<AActor>> TargetPosCollisionCheckClasses;

	UPROPERTY(BlueprintReadWrite, Category = "EnvState")
		bool b_all_agent_done;
	UPROPERTY(BlueprintReadWrite, Category = "EnvState")
		int32 num_left_new_agent;
	UPROPERTY(BlueprintReadWrite, Category = "EnvState")
		int32 newest_agent_index;

	UPROPERTY(BlueprintReadWrite, Category = "ImplicitEnv")
		class UTexture2D* implicit_env_texture;
	UPROPERTY(BlueprintReadWrite, Category = "ImplicitEnv")
		TMap<FString, class UTexture2D*> implicit_env_texture_dict;
	UPROPERTY(BlueprintReadWrite, Category = "ImplicitEnv")
		TArray<FColor> implicit_env_pixels;
	UPROPERTY(BlueprintReadWrite, Category = "ImplicitEnv")
		FVector2D implicit_env_texture_size;
	UPROPERTY(BlueprintReadWrite, Category = "ImplicitEnv")
		TArray<float> implicit_env_grid;
	UPROPERTY(BlueprintReadWrite, Category = "ImplicitEnv")
		int32 implicit_env_grid_size;
	UPROPERTY(BlueprintReadWrite, Category = "ImplicitEnv")
		TArray<FVector2D> route_nodes;
	UPROPERTY(BlueprintReadWrite, Category = "ImplicitEnv")
		bool b_show_grid;
	UPROPERTY(BlueprintReadWrite, Category = "ImplicitEnv")
		float grid_abs_size_x;
	UPROPERTY(BlueprintReadWrite, Category = "ImplicitEnv")
		float grid_abs_size_y;
	UPROPERTY(BlueprintReadWrite, Category = "ImplicitEnv")
		int32 grid_num_x;
	UPROPERTY(BlueprintReadWrite, Category = "ImplicitEnv")
		int32 grid_num_y;
	UPROPERTY(BlueprintReadWrite, Category = "ImplicitEnv")
		int32 nav_points_sample_interval;
	

public:	
	// Sets default values for this actor's properties
	AEnvBase();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	UFUNCTION(BlueprintCallable, Category = "RL")
		FString step(FS_Action actions);

	UFUNCTION(BlueprintCallable, Category = "RL")
		void sense();

	UFUNCTION(BlueprintCallable, Category = "RL")
		TArray<FS_FloatArray> get_observations();

	UFUNCTION(BlueprintCallable, Category = "RL")
		TArray<float> get_rewards();

	UFUNCTION(BlueprintCallable, Category = "RL")
		TArray<bool> get_dones();

	UFUNCTION(BlueprintCallable, Category = "RL")
		bool get_next_state(TArray<bool> dones, TArray<FS_FloatArray>& next_state);

	UFUNCTION(BlueprintCallable, Category = "RL")
		void get_exist_agents_indexes_and_locs(TArray<int32>& indexes, TArray<FVector2D>& positions);

	UFUNCTION(BlueprintCallable, Category = "RL")
		FString get_step_return();

	UFUNCTION(BlueprintCallable, Category = "RL")
		FString reset(FS_EnvConfig new_env_config);

	UFUNCTION(BlueprintNativeEvent, Category = "RL")
		void add_new_agents();
		virtual void add_new_agents_Implementation();

	UFUNCTION(BlueprintCallable, Category = "RL")
		class AAgentRL* add_new_agent_random(FS_AgentConfig agent_config);

	UFUNCTION(BlueprintCallable, Category = "RL")
		FVector2D generate_new_random_position(float agent_radius, float new_implicit_env_value_threshold);

	UFUNCTION(BlueprintCallable, Category = "RL")
		FVector2D generate_new_random_target_position(FVector2D agent_position, float agent_radius, float new_implicit_env_value_threshold);

	UFUNCTION(BlueprintCallable, Category = "ImplictEnv")
		void init_implicit_env_texture();

	UFUNCTION(BlueprintCallable, Category = "ImplictEnv")
		float get_implicit_env_pixel_value(FVector2D position);

	UFUNCTION(BlueprintCallable, Category = "ImplictEnv")
		TArray<FVector2D> find_nav_points(FVector2D start, FVector2D goal);

	UFUNCTION(BlueprintPure, Category = "ImplictEnv")
		FVector2D pos_to_grid_id(FVector2D position) const;

	UFUNCTION(BlueprintPure, Category = "ImplictEnv")
		FVector2D grid_id_to_pos(FVector2D position) const;

	UFUNCTION(BlueprintCallable, BlueprintImplementableEvent, Category = "RL")
		void update_env_vis();

};
