// Fill out your copyright notice in the Description page of Project Settings.


#include "EnvBase.h"
#include "AgentRL.h"
#include "Engine/Texture2D.h"
#include "TextureLib.h"
#include "Kismet/KismetSystemLibrary.h"

#include "Serialization/JsonWriter.h"
#include "Serialization/JsonSerializer.h"
#include "Dom/JsonObject.h"
#include "JsonObjectConverter.h"

// Sets default values
AEnvBase::AEnvBase()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
	env_size = FVector2D(30.f, 30.f);
	implicit_env_grid_size = 16; // Grid size (unit of pixel)
	nav_points_sample_interval = 3;
	FMath::RandInit(666);
}

// Called when the game starts or when spawned
void AEnvBase::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void AEnvBase::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

FString AEnvBase::step(FS_Action actions)
{
	for (int32 i = 0; i < actions.actions.Num(); i++)
	{
		agents[i]->update(actions.actions[i].floats);
		agents[i]->sense();
		agents[i]->reward_cur_step = agents[i]->get_reward();
		agents[i]->reward_cnt = agents[i]->reward_cur_step;
	}
	for (int32 i = 0; i < env_config.action_interval_step_num - 1; i++)
	{
		for (AAgentRL* agent : agents)
		{
			if (agent->bDone) continue;
			agent->update_without_action();
			agent->sense();
			agent->reward_cur_step = agent->get_reward();
			agent->reward_cnt += agent->reward_cur_step;
		}
	}
	return get_step_return();
}

void AEnvBase::sense()
{
	for (AAgentRL* agent : agents)
	{
		agent->sense();
	}
}

TArray<FS_FloatArray> AEnvBase::get_observations()
{
	TArray<FS_FloatArray> observations_temp;
	observations_temp.Empty();
	for (AAgentRL* agent : agents)
	{
		observations_temp.Add(agent->get_observation());
	}
	return observations_temp;
}

TArray<float> AEnvBase::get_rewards()
{
	TArray<float> rewards_temp;
	rewards_temp.Empty();
	for (AAgentRL* agent : agents)
	{
		rewards_temp.Add(agent->reward_cnt);
	}
	return rewards_temp;
}

TArray<bool> AEnvBase::get_dones()
{
	TArray<bool> dones_temp;
	dones_temp.Empty();
	for (AAgentRL* agent : agents)
	{
		dones_temp.Add(agent->bDone);
	}
	return dones_temp;
}

bool AEnvBase::get_next_state(TArray<bool> dones, TArray<FS_FloatArray>& next_state)
{
	// Check done agents
	int32 done_cnt = 0;
	TArray<int32> done_indexes;
	done_indexes.Empty();
	next_state.Empty();
	for (int32 i = 0; i < dones.Num(); i++)
	{
		if (!dones[i]) continue;
		done_cnt++;
		done_indexes.Add(i);
	}

	// Not any agent done
	if (done_cnt == 0)
	{
		return false;
	}

	// Replace agents array
	for (int32 done_index : done_indexes)
	{
		if (IsValid(agents[done_index])) agents[done_index]->Destroy();
	}
	new_agents.Empty();
	for (AAgentRL* agent : agents)
	{
		if (IsValid(agent))
		{
			new_agents.Add(agent);
		}
	}
	agents = new_agents;

	//Add new agents
	if (num_left_new_agent > 0)
	{
		int32 new_agent_num = FMath::Min(num_left_new_agent, done_cnt);
		for (int32 i = 0; i < new_agent_num; i++)
		{	
			FS_AgentConfig agent_config = FS_AgentConfig(
				0,
				FVector2D(),
				FVector2D(),
				env_config.agent_radius,
				env_config.num_rays,
				env_config.max_ray_distance,
				env_config.ray_vision_lamda,
				env_config.max_speed_orientation,
				env_config.max_speed_perpendicular,
				env_config.max_acceleration,
				env_config.max_angular_velocity,
				env_config.dt,
				env_config.max_alive_time,
				env_config.arrive_range_scale,
				env_config.max_time_overlaped_with_obstacle,
				env_config.implicit_env_obs_range,
				env_config.implicit_env_value_threshold,
				env_config.use_nav_point,
				env_config.update_nav_point_threshold,
				reward_config
			);
			agents.Add(add_new_agent_random(agent_config));
		}
		num_left_new_agent -= new_agent_num;
	}

	// Sense again
	sense();
	next_state = get_observations();
	if (next_state.Num() == 0)
	{
		b_all_agent_done = true;
	}

	return true;
}

void AEnvBase::get_exist_agents_indexes_and_locs(TArray<int32>& indexes, TArray<FVector2D>& positions)
{
	indexes.Empty();
	positions.Empty();
	for (AAgentRL* agent : agents)
	{
		indexes.Add(agent->index);
		positions.Add(agent->position);
	}
}

FString AEnvBase::get_step_return()
{	
	TArray<FS_FloatArray> observations = get_observations();
	TArray<float> rewards = get_rewards();
	TArray<bool> dones = get_dones();

	TArray<int32> before_update_agents_indexes;
	TArray<FVector2D> before_update_agents_locs;
	get_exist_agents_indexes_and_locs(before_update_agents_indexes, before_update_agents_locs);

	TArray<FS_FloatArray> next_state;
	bool has_done = get_next_state(dones, next_state);
	if (!has_done) next_state = observations;

	TArray<int32> exist_agents_indexes;
	TArray<FVector2D> exist_agents_locs;
	get_exist_agents_indexes_and_locs(exist_agents_indexes, exist_agents_locs);

	FS_StepReturn step_return = FS_StepReturn(
		observations,
		rewards,
		dones,
		next_state,
		b_all_agent_done,
		num_left_new_agent,
		before_update_agents_indexes,
		exist_agents_indexes,
		before_update_agents_locs,
		exist_agents_locs
	);
	FString OutputString;

	// TODO: Complete struct to json string
	FJsonObjectConverter::UStructToJsonObjectString(step_return, OutputString);

	return OutputString;
}

FString AEnvBase::reset(FS_EnvConfig new_env_config)
{
	// TODO: Finish reset function
	env_config = new_env_config;
	env_size = FVector2D(env_config.width, env_config.height);
	num_agents = env_config.num_agents;
	num_left_new_agent = env_config.num_new_agents_per_episode;
	b_all_agent_done = false;
	newest_agent_index = 0;

	update_env_vis();
	init_implicit_env_texture();

	for (AAgentRL* agent : agents)
	{
		agent->Destroy();
	}
	agents.Empty();

	add_new_agents();

	sense();
	TArray<FS_FloatArray> obs = get_observations();

	TArray<int32> indexes;
	TArray<FVector2D> locs;
	get_exist_agents_indexes_and_locs(indexes, locs);


	//TODO: struct to reset return string
	FString ReturnString;
	FS_ResetReturn reset_return = FS_ResetReturn(obs, locs);
	FJsonObjectConverter::UStructToJsonObjectString(reset_return, ReturnString);
	return ReturnString;
}

void AEnvBase::add_new_agents_Implementation()
{
	for (int32 i = 0; i < num_agents; i++)
	{
		FS_AgentConfig agent_config = FS_AgentConfig(
			0,
			FVector2D(),
			FVector2D(),
			env_config.agent_radius,
			env_config.num_rays,
			env_config.max_ray_distance,
			env_config.ray_vision_lamda,
			env_config.max_speed_orientation,
			env_config.max_speed_perpendicular,
			env_config.max_acceleration,
			env_config.max_angular_velocity,
			env_config.dt,
			env_config.max_alive_time,
			env_config.arrive_range_scale,
			env_config.max_time_overlaped_with_obstacle,
			env_config.implicit_env_obs_range,
			env_config.implicit_env_value_threshold,
			env_config.use_nav_point,
			env_config.update_nav_point_threshold,
			reward_config
		);
		agents.Add(add_new_agent_random(agent_config));
	}
}

AAgentRL* AEnvBase::add_new_agent_random(FS_AgentConfig agent_config)
{
	FVector2D new_position = generate_new_random_position(env_config.agent_radius, env_config.implicit_agent_spawn_threshold);
	FVector2D new_target_position = generate_new_random_target_position(new_position, env_config.agent_radius, env_config.implicit_agent_target_threshold);

	agent_config.index = newest_agent_index;
	agent_config.init_position = new_position;
	agent_config.target_position = new_target_position;

	AAgentRL* new_agent;
	if (AgentRLClass)
		new_agent = GetWorld()->SpawnActor<AAgentRL>(AgentRLClass, FVector(new_position, 0.f) * 100.f, FRotator());
	else
		new_agent = GetWorld()->SpawnActor<AAgentRL>(FVector(new_position, 0.f) * 100.f, FRotator());
	new_agent->env = this;
	new_agent->init_agent_rl(agent_config);
	newest_agent_index++;
	
	return new_agent;
}

FVector2D AEnvBase::generate_new_random_position(float agent_radius, float new_implicit_env_value_threshold)
{	
	bool position_valid = false;
	FVector2D new_position_temp;
	while (!position_valid)
	{
		// Generate position
		new_position_temp = FVector2D(
			FMath::FRandRange(agent_radius, env_size.X - agent_radius),
			FMath::FRandRange(agent_radius, env_size.Y - agent_radius)
		);
		position_valid = true;


		// Check implicit env value
		if (get_implicit_env_pixel_value(new_position_temp) < new_implicit_env_value_threshold)
		{
			position_valid = false;
		}
		else
		{
			// Check hit result
			FHitResult hit_result;
			TArray<AActor*> ignore_actors;
			bool check_hit = UKismetSystemLibrary::SphereTraceSingle(GetWorld(), FVector(new_position_temp, 10.f) * 100.f, FVector(new_position_temp, -10.f) * 100.f, agent_radius * 100.f, TraceTypeQuery1, false, ignore_actors, EDrawDebugTrace::None, hit_result, true);
			if (check_hit)
			{
				AActor* hit_actor = hit_result.GetActor();
				UClass* HitActorClass = hit_actor->GetClass();
				// Check if the hit actor's class matches any class in the array
				for (TSubclassOf<AActor> ActorClass : InitPosCollisionCheckClasses)
				{
					if (HitActorClass->IsChildOf(ActorClass))
					{
						position_valid = false;
						break;
					}
				}
			}
		}
	}
	return new_position_temp;
}

FVector2D AEnvBase::generate_new_random_target_position(FVector2D agent_position, float agent_radius, float new_implicit_env_value_threshold)
{
	bool position_valid = false;
	FVector2D new_position_temp;
	while (!position_valid)
	{
		// Generate position
		new_position_temp = FVector2D(
			FMath::FRandRange(env_config.target_loc_margin * agent_radius, env_size.X - env_config.target_loc_margin * agent_radius),
			FMath::FRandRange(env_config.target_loc_margin * agent_radius, env_size.Y - env_config.target_loc_margin * agent_radius)
		);
		position_valid = true;


		// Check implicit env value
		if (get_implicit_env_pixel_value(new_position_temp) < new_implicit_env_value_threshold)
		{
			position_valid = false;
		}
		else if (FVector2D::Distance(agent_position, new_position_temp) < env_config.min_walking_distance)
		{
			position_valid = false;
		}
		else
		{
			// Check hit result
			FHitResult hit_result;
			TArray<AActor*> ignore_actors;
			bool check_hit = UKismetSystemLibrary::SphereTraceSingle(GetWorld(), FVector(new_position_temp, 10.f) * 100.f, FVector(new_position_temp, -10.f) * 100.f, agent_radius * 100.f, TraceTypeQuery1, false, ignore_actors, EDrawDebugTrace::None, hit_result, true);
			if (check_hit)
			{
				AActor* hit_actor = hit_result.GetActor();
				UClass* HitActorClass = hit_actor->GetClass();
				// Check if the hit actor's class matches any class in the array
				for (TSubclassOf<AActor> ActorClass : TargetPosCollisionCheckClasses)
				{
					if (HitActorClass->IsChildOf(ActorClass))
					{
						position_valid = false;
						break;
					}
				}
			}
		}		
	}
	return new_position_temp;
}

void AEnvBase::init_implicit_env_texture()
{
	UTexture2D** FoundTexture = implicit_env_texture_dict.Find(env_config.implicit_env_texture_name);
	if (FoundTexture)
	{
		implicit_env_texture = *FoundTexture;
		implicit_env_texture_size = UTextureLib::ExtractTextureData(implicit_env_texture, implicit_env_pixels);
		implicit_env_grid = UTextureLib::CreateTextureGrid(implicit_env_pixels, implicit_env_texture_size.X, implicit_env_texture_size.Y, implicit_env_grid_size);
		grid_abs_size_y = implicit_env_grid_size / implicit_env_texture_size.Y * env_size.Y;
		grid_abs_size_x = implicit_env_grid_size / implicit_env_texture_size.X * env_size.X;
		grid_num_x = implicit_env_texture_size.X / implicit_env_grid_size;
		grid_num_y = implicit_env_texture_size.Y / implicit_env_grid_size;
	}
}

float AEnvBase::get_implicit_env_pixel_value(FVector2D position)
{
	FColor pos_color = UTextureLib::GetPixelColorFromStoredData(
		implicit_env_pixels,
		implicit_env_texture_size.X,
		implicit_env_texture_size.Y,
		position.X / env_size.X,
		position.Y / env_size.Y);
	return (0.2989 * pos_color.R + 0.587 * pos_color.G + 0.114 * pos_color.B) / 255.0;
}

TArray<FVector2D> AEnvBase::find_nav_points(FVector2D start, FVector2D goal)
{	
	TArray<FVector2D> path_grid_nodes_temp = UTextureLib::FindPathInTexture(implicit_env_grid, grid_num_x, grid_num_y, pos_to_grid_id(start), pos_to_grid_id(goal));
	TArray<FVector2D> path_grid_nodes;
	path_grid_nodes.Empty();
	for (int32 i = 0; i < path_grid_nodes_temp.Num(); i++)
	{
		if (i % nav_points_sample_interval == 0)
		{
			path_grid_nodes.Add(grid_id_to_pos(path_grid_nodes_temp[i]));
		}
	}
	return path_grid_nodes;
}

FVector2D AEnvBase::pos_to_grid_id(FVector2D position) const
{
	FVector2D grid_id = (position / env_size).ClampAxes(0.0f, 1.0f) * FVector2D(grid_num_x-1, grid_num_y-1);
	grid_id.X = FMath::RoundToFloat(grid_id.X);
	grid_id.Y = FMath::RoundToFloat(grid_id.Y);

	return grid_id;
}

FVector2D AEnvBase::grid_id_to_pos(FVector2D grid_id) const
{
	return grid_id * FVector2D(grid_abs_size_x, grid_abs_size_y) + 0.5 * FVector2D(grid_abs_size_x, grid_abs_size_y);
}

