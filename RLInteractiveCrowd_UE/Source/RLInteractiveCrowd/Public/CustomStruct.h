#pragma once

#include "CoreMinimal.h"
#include "CustomStruct.generated.h"

USTRUCT(BlueprintType)
struct FS_FloatArray
{
    GENERATED_BODY()

public:
    FS_FloatArray() {}

    FS_FloatArray(const TArray<float>& InFloats)
    {
        floats = InFloats;
    }

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
        TArray<float> floats;
};

USTRUCT(BlueprintType)
struct FS_RewardConfig
{
    GENERATED_BODY()

public:
    // Default init
    FS_RewardConfig()
        : alive_penalty(2.0f),
        velocity_smooth_penalty(0.25f),
        collision_penalty(50.0f),
        target_approach_distance_reward(100.0f),
        target_away_distance_penalty_ratio(0.1f),
        nav_point_approach_distance_reward(50.0f),
        target_arrive_reward(500.0f),
        obstacle_penalty(1.0f),
        obstacle_time_coef(0.1f),
        implicit_env_penalty(50.0f),
        reward_weight_entity_low(1.0f),
        reward_weight_entity_up(1.0f),
        reward_weight_implicit_low(1.0f),
        reward_weight_implicit_up(1.0f)
    {}

    // Paramized init
    FS_RewardConfig(
        float InAlivePenalty,
        float InVelocitySmoothPenalty,
        float InCollisionPenalty,
        float InTargetApproachDistanceReward,
        float InTargetAwayDistancePenaltyRatio,
        float InNavPointApproachDistanceReward,
        float InTargetArriveReward,
        float InObstaclePenalty,
        float InObstacleTimeCoef,
        float InImplicitEnvPenalty,
        float InRewardWeightEntityLow,
        float InRewardWeightEntityUp,
        float InRewardWeightImplicitLow,
        float InRewardWeightImplicitUp)
        : alive_penalty(InAlivePenalty),
        velocity_smooth_penalty(InVelocitySmoothPenalty),
        collision_penalty(InCollisionPenalty),
        target_approach_distance_reward(InTargetApproachDistanceReward),
        target_away_distance_penalty_ratio(InTargetAwayDistancePenaltyRatio),
        nav_point_approach_distance_reward(InNavPointApproachDistanceReward),
        target_arrive_reward(InTargetArriveReward),
        obstacle_penalty(InObstaclePenalty),
        obstacle_time_coef(InObstacleTimeCoef),
        implicit_env_penalty(InImplicitEnvPenalty),
        reward_weight_entity_low(InRewardWeightEntityLow),
        reward_weight_entity_up(InRewardWeightEntityUp),
        reward_weight_implicit_low(InRewardWeightImplicitLow),
        reward_weight_implicit_up(InRewardWeightImplicitUp)
    {}

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float alive_penalty;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float velocity_smooth_penalty;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float collision_penalty;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float target_approach_distance_reward;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float target_away_distance_penalty_ratio;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float nav_point_approach_distance_reward;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float target_arrive_reward;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float obstacle_penalty;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float obstacle_time_coef;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float implicit_env_penalty;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float reward_weight_entity_low;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float reward_weight_entity_up;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float reward_weight_implicit_low;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float reward_weight_implicit_up;
};

USTRUCT(BlueprintType)
struct FS_EnvConfig
{
    GENERATED_BODY()

public:
    FS_EnvConfig()
        : env_type(TEXT("DefaultEnvType")),
        env_title(TEXT("DefaultEnvTitle")),
        width(40.0f),
        height(40.0f),
        implicit_env_texture_name(TEXT("DefaultTexture")),
        num_agents(30),
		target_loc_margin(1.0f),
        agent_radius(0.5f),
        num_rays(50),
        max_ray_distance(10.0f),
        ray_vision_lamda(0.2f),
        max_speed_orientation(1.6f),
        max_speed_perpendicular(0.5f),
        max_acceleration(2.0f),
        max_angular_velocity(0.25*PI),
        dt(0.04f),
        action_interval_step_num(4),
        min_walking_distance(10.0f),
        max_alive_time(100.0f),
		arrive_range_scale(1.0f),
        max_time_overlaped_with_obstacle(2.0f),
        implicit_env_obs_range(FVector2D(10.f, 10.f)),
        implicit_env_value_threshold(0.9f),
        implicit_agent_spawn_threshold(0.9f),
        implicit_agent_target_threshold(0.9f),
		use_nav_point(false),
        update_nav_point_threshold(5.0f),
        num_new_agents_per_episode(200),
        num_user_agents_chase(0),
        num_user_agents_random(0),
        user_agent_max_speed_orientation(1.6f),
        user_agent_max_speed_perpendicular(0.5f),
        user_agent_max_acceleration(2.0f),
        user_agent_max_angular_velocity(0.25*PI),
        num_obstacles(0)
    {}

    FS_EnvConfig(
        FString InEnvType,
        FString InEnvTitle,
        float InWidth,
        float InHeight,
        FString InImplicitEnvTextureName,
        int32 InNumAgents,
        float InAgentRadius,
        int32 InNumRays,
		float InTargetLocMargin,
        float InMaxRayDistance,
        float InRayVisionLamda,
        float InMaxSpeedOrientation,
        float InMaxSpeedPerpendicular,
        float InMaxAcceleration,
        float InMaxAngularVelocity,
        float InDt,
        int32 InActionIntervalStepNum,
        float InMinWalkingDistance,
        float InMaxAliveTime,
		float InArriveRangeScale,
        float InMaxTimeOverlapedWithObstacle,
        FVector2D InImplicitEnvObsRange,
        float InImplicitEnvValueThreshold,
        float InImplicitAgentSpawnThreshold,
        float InImplicitAgentTargetThreshold,
		bool InUseNavPoint,
        float InUpdateNavPointThreshold,
        int32 InNumNewAgentsPerEpisode,
        int32 InNumUserAgentsChase,
        int32 InNumUserAgentsRandom,
        float InUserAgentMaxSpeedOrientation,
        float InUserAgentMaxSpeedPerpendicular,
        float InUserAgentMaxAcceleration,
        float InUserAgentMaxAngularVelocity,
        int32 InNumObstacles)
        : env_type(InEnvType),
        env_title(InEnvTitle),
        width(InWidth),
        height(InHeight),
        implicit_env_texture_name(InImplicitEnvTextureName),
        num_agents(InNumAgents),
        target_loc_margin(InTargetLocMargin),
        agent_radius(InAgentRadius),
        num_rays(InNumRays),
        max_ray_distance(InMaxRayDistance),
        ray_vision_lamda(InRayVisionLamda),
        max_speed_orientation(InMaxSpeedOrientation),
        max_speed_perpendicular(InMaxSpeedPerpendicular),
        max_acceleration(InMaxAcceleration),
        max_angular_velocity(InMaxAngularVelocity),
        dt(InDt),
        action_interval_step_num(InActionIntervalStepNum),
        min_walking_distance(InMinWalkingDistance),
        max_alive_time(InMaxAliveTime),
		arrive_range_scale(InArriveRangeScale),
        max_time_overlaped_with_obstacle(InMaxTimeOverlapedWithObstacle),
        implicit_env_obs_range(InImplicitEnvObsRange),
        implicit_env_value_threshold(InImplicitEnvValueThreshold),
        implicit_agent_spawn_threshold(InImplicitAgentSpawnThreshold),
        implicit_agent_target_threshold(InImplicitAgentTargetThreshold),
        use_nav_point(InUseNavPoint),
        update_nav_point_threshold(InUpdateNavPointThreshold),
        num_new_agents_per_episode(InNumNewAgentsPerEpisode),
        num_user_agents_chase(InNumUserAgentsChase),
        num_user_agents_random(InNumUserAgentsRandom),
        user_agent_max_speed_orientation(InUserAgentMaxSpeedOrientation),
        user_agent_max_speed_perpendicular(InUserAgentMaxSpeedPerpendicular),
        user_agent_max_acceleration(InUserAgentMaxAcceleration),
        user_agent_max_angular_velocity(InUserAgentMaxAngularVelocity),
        num_obstacles(InNumObstacles)
    {}

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EnvConfig")
        FString env_type;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EnvConfig")
        FString env_title;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EnvConfig")
        float width;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EnvConfig")
        float height;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EnvConfig")
        FString implicit_env_texture_name;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EnvConfig")
        int32 num_agents;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EnvConfig")
        float target_loc_margin;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float agent_radius;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        int32 num_rays;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float max_ray_distance;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float ray_vision_lamda;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float max_speed_orientation;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float max_speed_perpendicular;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float max_acceleration;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float max_angular_velocity;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float dt;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        int32 action_interval_step_num;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float min_walking_distance;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float max_alive_time;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float arrive_range_scale;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float max_time_overlaped_with_obstacle;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        FVector2D implicit_env_obs_range;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EnvConfig")
        float implicit_env_value_threshold;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EnvConfig")
        float implicit_agent_spawn_threshold;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EnvConfig")
        float implicit_agent_target_threshold;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EnvConfig")
        bool use_nav_point;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EnvConfig")
        float update_nav_point_threshold;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EnvConfig")
        int32 num_new_agents_per_episode;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "UserAgentConfig")
        int32 num_user_agents_chase;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "UserAgentConfig")
        int32 num_user_agents_random;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "UserAgentConfig")
        float user_agent_max_speed_orientation;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "UserAgentConfig")
        float user_agent_max_speed_perpendicular;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "UserAgentConfig")
        float user_agent_max_acceleration;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "UserAgentConfig")
        float user_agent_max_angular_velocity;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "EnvConfig")
        int32 num_obstacles;
};

USTRUCT(BlueprintType)
struct FS_AgentConfig
{
    GENERATED_BODY()

public:
    // Default init
    FS_AgentConfig()
        : index(0),
        init_position(FVector2D::ZeroVector),
        target_position(FVector2D::ZeroVector),
        radius(0.5f),
        num_rays(50),
        max_ray_distance(10.0f),
        ray_vision_lamda(0.2f),
        max_speed_orientation(1.6f),
        max_speed_perpendicular(0.5f),
        max_acceleration(2.0f),
        max_angular_velocity(0.25 * PI),
        dt(0.04f),
        max_alive_time(100.f),
		arrive_range_scale(1.0f),
        max_time_overlaped_with_obstacle(2.0f),
        implicit_env_obs_range(FVector2D(10.f, 10.f)),
        implicit_env_value_threshold(0.9f),
		use_nav_point(false),
        update_nav_point_threshold(5.0f),
        reward_config(FS_RewardConfig())
    {}

    // Paramized init
    FS_AgentConfig(int32 InIndex,
        FVector2D InInitPosition,
        FVector2D InTargetPosition,
        float InRadius,
        int32 InNumRays,
        float InMaxRayDistance,
        float InRayVisionLamda,
        float InMaxSpeedOrientation,
        float InMaxSpeedPerpendicular,
        float InMaxAcceleration,
        float InMaxAngularVelocity,
        float InDt,
        float InMaxAliveTime,
		float InArriveRangeScale,
        float InMaxTimeOverlapedWithObstacle,
        FVector2D InImplicitEnvObsRange,
        float InImplicitEnvValueThreshold,
		bool InUseNavPoint,
        float InUpdateNavPointThreshold,
        FS_RewardConfig InRewardConfig)
        : index(InIndex),
        init_position(InInitPosition),
        target_position(InTargetPosition),
        radius(InRadius),
        num_rays(InNumRays),
        max_ray_distance(InMaxRayDistance),
        ray_vision_lamda(InRayVisionLamda),
        max_speed_orientation(InMaxSpeedOrientation),
        max_speed_perpendicular(InMaxSpeedPerpendicular),
        max_acceleration(InMaxAcceleration),
        max_angular_velocity(InMaxAngularVelocity),
        dt(InDt),
        max_alive_time(InMaxAliveTime),
		arrive_range_scale(InArriveRangeScale),
        max_time_overlaped_with_obstacle(InMaxTimeOverlapedWithObstacle),
        implicit_env_obs_range(InImplicitEnvObsRange),
        implicit_env_value_threshold(InImplicitEnvValueThreshold),
        use_nav_point(InUseNavPoint),
        update_nav_point_threshold(InUpdateNavPointThreshold),
        reward_config(InRewardConfig)
    {}
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        int32 index;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        FVector2D init_position;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        FVector2D target_position;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float radius;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        int32 num_rays;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float max_ray_distance;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float ray_vision_lamda;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float max_speed_orientation;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float max_speed_perpendicular;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float max_acceleration;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float max_angular_velocity;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float dt;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float max_alive_time;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float arrive_range_scale;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float max_time_overlaped_with_obstacle;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        FVector2D implicit_env_obs_range;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float implicit_env_value_threshold;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        bool use_nav_point;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        float update_nav_point_threshold;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AgentConfig")
        FS_RewardConfig reward_config;
};

USTRUCT(BlueprintType)
struct FS_Action
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
        TArray<FS_FloatArray> actions;
};

USTRUCT(BlueprintType)
struct FS_StepReturn
{
    GENERATED_BODY()

public:
    FS_StepReturn()
        : all_agent_done(false),
        num_left_new_agent(0)
    {}

    FS_StepReturn(
        const TArray<FS_FloatArray>& InObs,
        const TArray<float>& InRewards,
        const TArray<bool>& InDones,
        const TArray<FS_FloatArray>& InNextState,
        bool InAllAgentDone,
        int32 InNumLeftNewAgent,
        const TArray<int32>& InBeforeUpdateAgentIndexes,
        const TArray<int32>& InExistAgentIndexes,
        const TArray<FVector2D>& InBeforeUpdateAgentLocs,
        const TArray<FVector2D>& InExistAgentLocs)
        : obs(InObs),
        rewards(InRewards),
        dones(InDones),
        next_state(InNextState),
        all_agent_done(InAllAgentDone),
        num_left_new_agent(InNumLeftNewAgent),
        before_update_agent_indexes(InBeforeUpdateAgentIndexes),
        exist_agent_indexes(InExistAgentIndexes),
        before_update_agent_locs(InBeforeUpdateAgentLocs),
        exist_agent_locs(InExistAgentLocs)
    {}

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
        TArray<FS_FloatArray> obs;
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
        TArray<float> rewards;
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
        TArray<bool> dones;
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
        TArray<FS_FloatArray> next_state;
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
        bool all_agent_done;
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
        int32 num_left_new_agent;
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
        TArray<int32> before_update_agent_indexes;
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
        TArray<int32> exist_agent_indexes;
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
        TArray<FVector2D> before_update_agent_locs;
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
        TArray<FVector2D> exist_agent_locs;
};

USTRUCT(BlueprintType)
struct FS_ResetReturn
{
    GENERATED_BODY()

public:
    FS_ResetReturn()
    {}

    FS_ResetReturn(
        const TArray<FS_FloatArray>& InObs,
        const TArray<FVector2D>& InExistAgentLocs)
        : obs(InObs),
        exist_agent_locs(InExistAgentLocs)
    {}

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
        TArray<FS_FloatArray> obs;
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
        TArray<FVector2D> exist_agent_locs;
};