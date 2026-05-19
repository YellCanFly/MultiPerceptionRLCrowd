// Fill out your copyright notice in the Description page of Project Settings.


#include "CommonLib.h"
#include "Components/TimelineComponent.h"


bool UCommonLib::AddEventToTimeline(UTimelineComponent* Timeline, float Time, FName EventName, UObject* InObject)
{
	if (Timeline == NULL) return false;
	if (Time > Timeline->GetTimelineLength()) return false;
	if (!InObject->FindFunction(EventName)) return false;
	FOnTimelineEvent onTimelineEvent;
	onTimelineEvent.BindUFunction(InObject, EventName);
	Timeline->AddEvent(Time, onTimelineEvent);
	return true;
}

bool UCommonLib::BindFinishEventToTimeline(UTimelineComponent* Timeline, FName EventName, UObject* InObject)
{
	if (Timeline == NULL || InObject == NULL) return false;
	// Bind time line finish call back event
	FOnTimelineEvent onTimelineFinishCallback;
	onTimelineFinishCallback.BindUFunction(InObject, EventName);
	Timeline->SetTimelineFinishedFunc(onTimelineFinishCallback);
	return true;
}

void UCommonLib::UnbindFinishEventToTimeline(UTimelineComponent* Timeline, UObject* InObject)
{
	FOnTimelineEvent onTimelineFinishCallback;
	onTimelineFinishCallback.BindUFunction(InObject, FName(TEXT("")));
	Timeline->SetTimelineFinishedFunc(onTimelineFinishCallback);
}

void UCommonLib::CallFunctionByName(FName FuncName, UObject* InObject)
{
	UFunction* TargetFunc = InObject->FindFunction(FuncName);
	if (TargetFunc == nullptr) return;
	InObject->ProcessEvent(TargetFunc, NULL);
}