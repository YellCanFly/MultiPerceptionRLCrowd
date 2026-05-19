// Fill out your copyright notice in the Description page of Project Settings.


#include "AgentRL.h"
#include "EnvBase.h"
#include "Math/UnrealMathUtility.h"
#include "Components/SphereComponent.h"


AAgentRL::AAgentRL(const FObjectInitializer& ObjectInitializer)
{
	draw_trace_type = EDrawDebugTrace::None;


	// Bind the overlap events
	SphereCollision->OnComponentBeginOverlap.AddDynamic(this, &AAgentRL::OnOverlapBegin);
	SphereCollision->OnComponentEndOverlap.AddDynamic(this, &AAgentRL::OnOverlapEnd);
}

void AAgentRL::init_agent_rl(FS_AgentConfig agent_config)
{
	//Init from agent confit
	this->index = agent_config.index;
	this->position = agent_config.init_position;
	this->target_position = agent_config.target_position;
	this->radius = agent_config.radius;
	this->num_rays = agent_config.num_rays;
	this->max_ray_distance = agent_config.max_ray_distance;
	this->ray_vision_lamda = agent_config.ray_vision_lamda;
	this->max_speed_orientation = agent_config.max_speed_orientation;
	this->max_speed_perpendicular = agent_config.max_speed_perpendicular;
	this->max_acceleration = agent_config.max_acceleration;
	this->max_angular_velocity = agent_config.max_angular_velocity;
	this->dt = agent_config.dt;
	this->max_alive_time = agent_config.max_alive_time;
	this->arrive_range_scale = agent_config.arrive_range_scale;
	this->max_time_overlaped_with_obstacle = agent_config.max_time_overlaped_with_obstacle;
	this->implicit_env_obs_range = agent_config.implicit_env_obs_range;
	this->implicit_env_value_threshold = agent_config.implicit_env_value_threshold;
	this->use_nav_point = agent_config.use_nav_point;
	this->update_nav_point_threshold = agent_config.update_nav_point_threshold;
	this->reward_config = agent_config.reward_config;

	// Init value
	this->target_offset = 0.f;
	this->alive_time = 0.f;
	this->orientation = FMath::FRandRange(0.f, 2 * PI);
	this->velocity = FVector2D(
		FMath::FRandRange(this->max_speed_orientation * this->back_forward_velocity_ratiol, this->max_speed_orientation),
		FMath::FRandRange(-1.f * this->max_speed_perpendicular, this->max_speed_perpendicular)
	);
	this->reward_weight_entity = FMath::FRandRange(reward_config.reward_weight_entity_low, reward_config.reward_weight_entity_up);
	this->reward_weight_implicit = FMath::FRandRange(reward_config.reward_weight_implicit_low, reward_config.reward_weight_implicit_up);

	// Init ray-casting
	this->ray_distances.Empty();
	this->max_ray_distances.Empty();
	for (int i = 0; i < num_rays; i++)
	{	
		float theta = i * 2.0 * PI / (num_rays - 1);
		float max_ray_distance_i = max_ray_distance * (ray_vision_lamda + (1 - ray_vision_lamda) * 0.5 * (1 + FMath::Cos(theta)));
		ray_distances.Add(max_ray_distance_i);
		max_ray_distances.Add(max_ray_distance_i);
	}

	// Init navigation points
	if (IsValid(env))
	{
		this->nav_points = env->find_nav_points(position, target_position);
		this->current_nav_point_index = 0;
	}

	// Init visulization
	this->UpdateVisualization();
}

void AAgentRL::update(TArray<float> action)
{	
	// Extract actions
	this->acceleration = FVector2D(action[0], action[1]);
	this->angular_velocity = action[2];

	// Update Velocity
	this->velocity_last_step = this->velocity;
	this->velocity = this->velocity + this->dt * this->acceleration;
	this->velocity.X = FMath::Clamp(this->velocity.X, max_speed_orientation * back_forward_velocity_ratiol, max_speed_orientation);
	this->velocity.Y = FMath::Clamp(this->velocity.Y, -1.f * max_speed_perpendicular, max_speed_perpendicular);

	// Update Position
	this->position_last_step = this->position;
	FVector2D move_offset = velocity_to_absolute(velocity) * dt;
	float target_distance_last_step = (target_position - position).Length();
	this->position += move_offset;
	this->position.X = FMath::Clamp(this->position.X, radius, env->env_size.X - radius);
	this->position.Y = FMath::Clamp(this->position.Y, radius, env->env_size.Y - radius);

	// Update target distance
	FVector2D agent_to_target_direction = (target_position - position).GetSafeNormal();
	this->agent_to_target_distance = (target_position - position).Length();
	this->target_approach_distance = target_distance_last_step - agent_to_target_distance;
	this->target_offset = move_offset.Dot(agent_to_target_direction);

	// Update orientation
	this->orientation += angular_velocity * dt;
	
	// Update time record
	this->alive_time += dt;
	if (is_overlaped_with_obstacle)
	{
		this->overlap_obstacle_time += dt;
	}

	// Update navigation point and approach distance
	nav_point_approach_diatance = 0.0f;
	if (use_nav_point)
	{
		for (int32 i = current_nav_point_index; i < nav_points.Num(); i++)
		{
			if (FVector2D::Distance(nav_points[i], position) < update_nav_point_threshold)
			{
				current_nav_point_index = i + 1;
			}
		}
		if (current_nav_point_index >= nav_points.Num())
		{
			use_nav_point = false;
		}
		else
		{
			nav_point_approach_diatance = FVector2D::Distance(nav_points[current_nav_point_index], position_last_step) - FVector2D::Distance(nav_points[current_nav_point_index], position);
		}
	}

	// Update terminate sign
	this->bDone = is_done();

	// Update visulization
	UpdateVisualization();
	
}

void AAgentRL::update_without_action()
{
	// Update Position
	this->position_last_step = this->position;
	FVector2D move_offset = velocity_to_absolute(velocity) * dt;
	float target_distance_last_step = (target_position - position).Length();
	this->position += move_offset;
	this->position.X = FMath::Clamp(this->position.X, radius, env->env_size.X - radius);
	this->position.Y = FMath::Clamp(this->position.Y, radius, env->env_size.Y - radius);

	// Update target distance
	FVector2D agent_to_target_direction = (target_position - position).GetSafeNormal();
	this->agent_to_target_distance = (target_position - position).Length();
	this->target_approach_distance = target_distance_last_step - agent_to_target_distance;
	this->target_offset = move_offset.Dot(agent_to_target_direction);

	// Update time record
	this->alive_time += dt;
	if (is_overlaped_with_obstacle)
	{
		this->overlap_obstacle_time += dt;
	}

	// Update navigation point and approach distance
	nav_point_approach_diatance = 0.0f;
	if (use_nav_point)
	{
		for (int32 i = current_nav_point_index; i < nav_points.Num(); i++)
		{
			if (FVector2D::Distance(nav_points[i], position) < update_nav_point_threshold)
			{
				current_nav_point_index = i + 1;
			}
		}
		if (current_nav_point_index >= nav_points.Num())
		{
			use_nav_point = false;
		}
		else
		{
			nav_point_approach_diatance = FVector2D::Distance(nav_points[current_nav_point_index], position_last_step) - FVector2D::Distance(nav_points[current_nav_point_index], position);
		}
	}

	// Update terminate sign
	this->bDone = is_done();

	// Update visulization
	this->UpdateVisualization();
}

void AAgentRL::sense()
{
	// Ray-casting sense (Entity perception)
	FVector2D orientation_direction = FVector2D(FMath::Cos(orientation), FMath::Sin(orientation));
	ray_object_relative_velocities.Empty();
	for (int32 i = 0; i < num_rays; i++)
	{
		float theta = i * 2.0 * PI / (num_rays - 1) + orientation;
		FVector2D ray_direction = FVector2D(FMath::Cos(theta), FMath::Sin(theta));
		//UE_LOG(LogTemp, Warning, TEXT("ID %d :ray_direction: %f, %f, %f, %f."), i, theta, orientation, ray_direction.X, ray_direction.Y);
		FHitResult HitResult;

		// Ö´ÐÐÏßÐÔ×·×Ù
		FVector trace_start = FVector(position + ray_direction * radius, 0.0f) * 100.f;
		FVector trace_end = FVector(position + ray_direction * (radius + max_ray_distances[i]), 0.0f) * 100.f;
		TArray<AActor*> IgnoreActors;
		IgnoreActors.Add(this);
		bool bHit = UKismetSystemLibrary::LineTraceSingle(GetWorld(), trace_start, trace_end, TraceTypeQuery1, false, IgnoreActors, draw_trace_type, HitResult, true);
		//UE_LOG(LogTemp, Warning, TEXT("ID %d :Hit Start: %f, %f; Hit End: %f, %f; Hit: %s."), i, trace_start.X, trace_start.Y, trace_end.X, trace_end.Y, bHit ? TEXT("true") : TEXT("false"));
		if (bHit)
		{	
			ray_distances[i] = HitResult.Distance * 0.01;
			AActor* HitActor = HitResult.GetActor();
			if (HitActor)
			{
				AAgentBase* hit_agent = Cast<AAgentBase>(HitActor);
				if (hit_agent)
				{
					ray_object_relative_velocities.Add(velocity_to_relative(velocity_to_absolute(hit_agent->velocity) - velocity_to_absolute(velocity)));
				}
				else
				{
					ray_object_relative_velocities.Add(FVector2D(0.f, 0.f));
				}
			}
			else
			{
				ray_object_relative_velocities.Add(FVector2D(0.f, 0.f));
			}
		}
		else
		{
			ray_distances[i] = max_ray_distances[i];
			ray_object_relative_velocities.Add(FVector2D(0.f, 0.f));
		}
	}

	// Collision distances sense (Entity perception)
	collision_distances.Empty();
	for (int32 i = 0; i < collision_agents.Num(); i++)
	{
		collision_distances.Add(FVector2D::Distance(collision_agents[i]->position, position));
	}

	// Implicit environment value sense (Implicit perception)
	implicit_env_value = 0.f;
	if (IsValid(env))
	{
		implicit_env_value = env->get_implicit_env_pixel_value(position);
	}
		
}

FS_FloatArray AAgentRL::get_observation()
{
	TArray<float> observations;
	observations.Empty();
	for (int32 i = 0; i < num_rays; i++)
	{
		observations.Add(ray_distances[i] / max_ray_distances[i]);
		observations.Add(ray_object_relative_velocities[i].X);
		observations.Add(ray_object_relative_velocities[i].Y);
	}

	observations.Add(reward_weight_entity);
	observations.Add(reward_weight_implicit);

	observations.Add(orientation);
	observations.Add(velocity.X);
	observations.Add(velocity.Y);
	FVector2D target_position_relative = position_to_relative(target_position);
	observations.Add(target_position_relative.X);
	observations.Add(target_position_relative.Y);

	FVector2D nav_point = use_nav_point ? nav_points[current_nav_point_index] : target_position;
	FVector2D nav_point_relativa = position_to_relative(nav_point);
	observations.Add(nav_point_relativa.X);
	observations.Add(nav_point_relativa.Y);

	return FS_FloatArray(observations);
}

float AAgentRL::get_reward()
{
	float reward_temp = 0.f;

	// Alive penalty
	reward_temp -= reward_config.alive_penalty;

	// Velocity smooth penalty
	reward_temp -= reward_config.velocity_smooth_penalty * FVector2D::Distance(velocity, velocity_last_step);

	// Collision agents penalty
	for (AAgentBase* coll_agent : collision_agents)
	{	
		// w_e * c * (exp(r1 + r2 - d) - 1)
		reward_temp -= reward_weight_entity * reward_config.collision_penalty * (FMath::Exp(coll_agent->radius + radius - FVector2D::Distance(coll_agent->position, position)) - 1);
	}

	// Target approach reward
	float approach_coef = target_approach_distance > 0 ? 1 : reward_config.target_away_distance_penalty_ratio;
	//reward_temp += reward_config.target_approach_distance_reward * target_approach_distance;
	reward_temp += reward_config.target_approach_distance_reward * target_approach_distance * approach_coef;

	// Nav point approach reward
	if (use_nav_point)
	{
		reward_temp += reward_config.nav_point_approach_distance_reward * FMath::Max(nav_point_approach_diatance, 0.f);
	}

	// Implicit env penalty
	//reward_temp -= reward_weight_implicit * reward_config.implicit_env_penalty * FMath::Max((implicit_env_value_threshold - implicit_env_value), 0.f);
	reward_temp -= reward_weight_implicit * reward_config.implicit_env_penalty * (implicit_env_value_threshold - implicit_env_value);

	// Arrive target point reward
	if (is_arrived_target())
	{
		reward_temp += reward_config.target_arrive_reward;
	}
	
	// Obstacle collision penalty
	if (is_overlaped_with_obstacle)
	{
		reward_temp -= reward_config.obstacle_penalty * FMath::Exp(reward_config.obstacle_time_coef * (overlap_obstacle_time / dt));
	}

	return reward_temp;
}

bool AAgentRL::is_done()
{	
	bool condition_1 = is_arrived_target();
	bool condition_2 = alive_time >= max_alive_time;
	bool condition_3 = overlap_obstacle_time > max_time_overlaped_with_obstacle;
	return condition_1 || condition_2 || condition_3;
}

bool AAgentRL::is_arrived_target()
{
	return agent_to_target_distance < arrive_range_scale * radius;
}

void AAgentRL::OnOverlapBegin(UPrimitiveComponent* OverlappedComp, AActor* OtherActor, UPrimitiveComponent* OtherComp, int32 OtherBodyIndex, bool bFromSweep, const FHitResult& SweepResult)
{
	AAgentBase* overlapped_agent = Cast<AAgentBase>(OtherActor);
	if (overlapped_agent)
	{
		collision_agents.Add(overlapped_agent);
	}
	else
	{
		if (OtherComp->GetCollisionObjectType() == ECollisionChannel::ECC_WorldStatic)
		{
			is_overlaped_with_obstacle = true;
		}
	}
}

void AAgentRL::OnOverlapEnd(UPrimitiveComponent* OverlappedComp, AActor* OtherActor, UPrimitiveComponent* OtherComp, int32 OtherBodyIndex)
{
	AAgentBase* overlapped_agent = Cast<AAgentBase>(OtherActor);
	if (overlapped_agent)
	{
		collision_agents.Remove(overlapped_agent);
	}
	else
	{
		if (OtherComp->GetCollisionObjectType() == ECollisionChannel::ECC_WorldStatic)
		{
			is_overlaped_with_obstacle = false;
			overlap_obstacle_time = 0.f;
		}
	}

}

void AAgentRL::UpdateVisualization_Implementation()
{
	Super::UpdateVisualization_Implementation();
	//UE_LOG(LogTemp, Warning, TEXT("C++ implementation of AAgentRL"));
	set_agent_location();
}
