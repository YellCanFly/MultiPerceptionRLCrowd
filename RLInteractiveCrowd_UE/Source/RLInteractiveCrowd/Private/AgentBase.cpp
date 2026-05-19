// Fill out your copyright notice in the Description page of Project Settings.


#include "AgentBase.h"
#include "Components/SphereComponent.h"
#include "Components/SceneComponent.h"

// Sets default values
AAgentBase::AAgentBase()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	radius = 0.5;
	back_forward_velocity_ratiol = -0.3125;

	Root = CreateDefaultSubobject<USceneComponent>(TEXT("RootComponent"));
	RootComponent = Root;

	SphereCollision = CreateDefaultSubobject<USphereComponent>(TEXT("SphereCollision"));
	SphereCollision->SetupAttachment(RootComponent);
	SphereCollision->SetSphereRadius(this->radius * 100.0f, true);

}

// Called when the game starts or when spawned
void AAgentBase::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void AAgentBase::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

FVector2D AAgentBase::velocity_to_absolute(FVector2D relative_velocity) const
{
	return FVector2D(FVector(relative_velocity, 0.f).RotateAngleAxisRad(this->orientation, FVector(0.f, 0.f, 1.f)));
}

FVector2D AAgentBase::velocity_to_relative(FVector2D absolute_velocity) const
{
	return FVector2D(FVector(absolute_velocity, 0.f).RotateAngleAxisRad(-1.0*this->orientation, FVector(0.f, 0.f, 1.f)));
}

FVector2D AAgentBase::position_to_relative(FVector2D absolute_position) const
{
	return FVector2D(FVector(absolute_position - this->position, 0.f).RotateAngleAxisRad(-1.0 * this->orientation, FVector(0.f, 0.f, 1.f)));
}

void AAgentBase::UpdateVisualization_Implementation()
{
	//UE_LOG(LogTemp, Warning, TEXT("C++ implementation of AgentBase"));
}

void AAgentBase::set_agent_location()
{
	this->SetActorLocation(100.0 * FVector(this->position, 0.f));
}

